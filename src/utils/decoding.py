"""Sequence to Coordinate decoding algorithms.

Various CTC and Seq2Seq decoding algorithms to produce the respective output
coordinates and/or sequences (depending on the nature of the algorithm).
"""

from __future__ import annotations

from typing import List, Optional, NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike


BLANK_CHARACTER: int = 0
NO_CHARACTER: int = -1
FIRST_ELEMENT: int = -1
MAX_WIDTH: int = 0


class Coordinate(NamedTuple):
    """Represents a 1D coordinate range for alignment."""

    x1: int
    x2: int


class Prediction:
    """Encapsulate a predicted sequence into a single object."""

    def __init__(
            self,
            coordinates: List[Coordinate | None],
            confidences: List[float],
            characters: List[int],
    ) -> None:
        """Create a Prediction object.

        :param coordinates: A list of Coordinate objects that represent the
        starting and ending pixels of a given object in the line.
        :param confidences: Confidence values for each coordinate tuple.
        :param characters: The character represented in each prediction step.
        """
        self._coordinates = coordinates
        self._confidences = confidences
        self._characters = characters

    def __iter__(
            self,
    ) -> Tuple[Coordinate | None, float, int]:
        """Iterate over Prediction elements.

        :returns: A Tuple containing a Coordinate object, a confidence value
        and the underlying character at each specific time step.
        """
        for pred, conf, char in zip(
                self._coordinates,
                self._confidences,
                self._characters
        ):
            yield pred, conf, char

    @staticmethod
    def from_ctc_decoding(
            char_indices: ArrayLike,
            gt_sequence: ArrayLike,
            ctc_matrix: ArrayLike,
            column_size: int,
    ) -> Prediction:
        """Create Prediction from the output of a PrefixTree CTC decoding.

        :param char_indices: Array containing the class of each CTC column
        w.r.t. the input ground truth sequence. Essentially, the index of the
        character in the ground truth sequence.
        :param gt_sequence: Array containing the ground truth sequence to align
        to.
        :param ctc_matrix: (sequence length, batch size, class confidence)
        matrix produced by a model.
        :param column_size: Size in pixels of each column in the CTC model.
        """
        coordinates = []
        confidences = []
        characters = []

        current_char = None
        start_coordinate = -1
        partial_confidences = []

        for ind, char_index in enumerate(char_indices):
            if char_index == -1:
                continue
            if current_char is None:
                current_char = char_index
                start_coordinate = ind * column_size
            else:
                # If a different char is found, save the previous one and reset
                if current_char != char_index:
                    characters.append(gt_sequence[current_char])
                    coordinates.append(
                        Coordinate(start_coordinate, ind * column_size)
                    )
                    confidences.append(np.array(partial_confidences).mean())

                    current_char = char_index
                    start_coordinate = ind * column_size
                    partial_confidences = []

                # Otherwise keep accumulating confidences
                else:
                    partial_confidences.append(
                        ctc_matrix[ind, gt_sequence[char_index]]
                    )
        if current_char is not None:
            characters.append(gt_sequence[current_char])
            coordinates.append(
                Coordinate(start_coordinate, len(ctc_matrix) * column_size)
            )
            confidences.append(np.array(partial_confidences).mean())

        coordinates = np.array(coordinates)
        confidences = np.array(confidences)
        characters = np.array(characters)

        return Prediction(coordinates, confidences, characters)


class PredictionGroup:
    """Aggregate (Ensemble) multiple predictions."""

    def __init__(
            self,
            predictions: List[Prediction],
            names: List[str],
    ) -> None:
        """Create PredictionGroup object.

        :param predictions: A list of Prediction objects of the same image
        line. They must contain the same number of elements, as they are
        produced from the same ground truth sequence as reference.
        :param names: A list of method names to keep track of results in
        an orderly fashion.
        """
        self._predictions = predictions
        self._names = names

    def find_anchors(
            self
    ) -> Prediction:
        """Compute high-consensus predictions.

        Given a set of predicted alignments, provide the main consensus
        elements among all methods. The idea is finding those predictions in
        which plenty of methods agree or whose properties make sense in the
        overall scheme of things (no huge characters, no missed elements, etc)

        :returns: A Prediction composed only of the anchor elements.
        """
        raise NotImplementedError


def decode_ctc(
    ctc_mat: ArrayLike,
    out_seq: ArrayLike,
    beam_width: int = MAX_WIDTH,
) -> ArrayLike:
    """Produce the most likely decoding of out_seq.

    Iterate over all possible transcriptions of each batch in a ctc model
    and produce the most likely decoding of the ground truth sequence.

    :param ctc_mat: (sequence length, batch size, class confidence) matrix
    produced by a model.
    :param out_seq: (batch size, sequence length) array with numbers as
    elements in the sequence (zero is strictly reserved for the blank symbol).
    """
    outputs = []
    ctc_mat = ctc_mat.transpose((1, 0, 2))
    batch_size, seqlen, classes = ctc_mat.target_shape

    for mat, transcript in zip(ctc_mat, out_seq):
        tree = PrefixTree(
            None,
            transcript,
            beam_width
        )
        outputs.append(tree.decode())
    return outputs


class PrefixNode:
    """Node within a prefix tree with the full parent nodes' transcription."""

    def __init__(
            self,
            character: int,
            char_index: int,
            parent: Optional[PrefixNode],
            confidence: float,
    ) -> None:
        """Construct a PrefixNode.

        :param character: Character pertaining to the current node.
        :param char_index: Index of the position of the character in the output
        string.
        :param parent: Pointer to the parent node if it exists.
        :param confidence: Cumulative log probability of the sequence after the
        inclusion of this node.
        """
        self._character = character
        self._char_index = char_index
        self._parent = parent
        self._confidence = confidence
        self._children = []

    @property
    def character(
            self
    ) -> int:
        """Produce the character associated to this node."""
        return self._character

    @character.setter
    def character(
            self,
            value: int,
    ) -> None:
        self._character = value

    @property
    def char_index(
            self,
    ) -> int:
        """Produce the character index associated to this node."""
        return self._char_index

    @char_index.setter
    def char_index(
            self,
            value: int,
    ) -> None:
        self._char_index = value

    @property
    def confidence(
            self,
    ) -> float:
        """Produce current log likelihood of the prefix tree."""
        return self._confidence

    @confidence.setter
    def confidence(
            self,
            value: float,
    ) -> float:
        self._confidence = value

    def expand(
            self,
            character: int,
            char_index: int,
            confidence: float,
    ) -> PrefixNode:
        """Add a PrefixNode child to the current node.

        :param character: Character pertaining to the current node.
        :param char_index: Index of the position of the character in the output
        string.
        :param confidence: Confidence value of the current node addition.

        :returns: Produced node added as child to the current one.
        """
        cumulative_confidence = self._confidence + np.log(confidence)

        node = PrefixNode(character, char_index, self, cumulative_confidence)
        self._children.append(node)

        return node

    def produce_sequence(
            self,
    ) -> ArrayLike:
        """Produce the sequence associated to this element of the prefix tree.

        :returns: Character index sequence associated to this element of the
        prefix tree. If the confidence is required, simply call
        PrefixNode.confidence.
        """
        node = self
        decoding = [node._char_index]

        while node._parent is not None:
            decoding.append(node._char_index)
            node = node._parent

        decoding.reverse()
        return np.array(decoding)


# TODO Implement this but using a dynamic programming algorithm (I *think* it
# is possible, but I am not sure. For the time being, beam decoding should
# yield optimum results anyway).
class PrefixTree:
    """Compute the highest log likelihood decoding of a given GT sequence."""

    def __init__(
            self,
            root_character: Optional[PrefixNode],
            output_sequence: ArrayLike,
            beam_width: int = MAX_WIDTH,
    ) -> None:
        """Create a PrefixTree for CTC decoding given the gt sequence.

        :param root_character: A PrefixNode object with the starting character
        to perform decoding.
        :param output_sequence: The gt sequence with character indices as
        elements of a Numpy array.
        :param beam_width: Number of decoded sequences to keep at a time step.
        """
        if root_character is None:
            root_character = PrefixNode(BLANK_CHARACTER, FIRST_ELEMENT, 0.0)

        self._root_character = root_character
        self._output_sequence = output_sequence
        self._beams: List[PrefixNode] = [self._root_character]
        self._beam_width = beam_width

    def decode(
            self,
            ctc_matrix: ArrayLike,
    ) -> ArrayLike:
        """Produce the most likely decoding of the gt sequence.

        :param ctc_matrix: A (SeqLength x NumClasses) numpy array with each
        character prediction's confidence at each time step.

        :returns: Highest likelihood character indices for the ground truth
        sequence.
        """
        for position, column in enumerate(ctc_matrix):
            level_nodes = []

            for node in self._beams:
                char_index = node._char_index
                character = node._character

                if not character == BLANK_CHARACTER:

                    # Add same character as current node into the expansion
                    level_nodes.append(node.expand(
                        self.output_sequence[char_index],
                        char_index,
                        column[character],
                    ))

                # Add blank character
                level_nodes.append(node.expand(
                    BLANK_CHARACTER,
                    char_index,
                    column[BLANK_CHARACTER],
                ))

                if char_index < len(self._output_sequence):
                    # Add next character
                    level_nodes.append(node.expand(
                        self.output_sequence[char_index + 1],
                        char_index + 1,
                        column[self.output_sequence[char_index + 1]],
                    ))

            level_nodes.sort(key=lambda x: x.confidence, reverse=True)

            if self._beam_width > 0:
                level_nodes[:self._beam_width]

            self._beams = level_nodes

        return self._beams[0].produce_sequence()


def decode_ctc_greedy(ctc_matrix: ArrayLike) -> List[int]:
    """Decode a CTC output matrix greedily and produce a sequence list.

    :param ctc_matrix: An output matrix from a CTC-like model. Should have
    shape S x C, where S is the sequence length and C is the number of classes.
    :returns: A list containing the decoded sequence.
    """
    maximum = ctc_matrix.argmax(axis=-1)
    last_char = BLANK_CHARACTER
    output = []
    for char in maximum:
        if char != BLANK_CHARACTER:
            if last_char == BLANK_CHARACTER or last_char != char:
                output.append(char)
                last_char = char
        else:
            last_char = BLANK_CHARACTER
    return output
