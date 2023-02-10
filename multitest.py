import time
from multiprocessing import Pool

    
def err(a):
    raise Exception


class A:
    def __init__(self):
        self.a = 1

    def foo(self):
        self.a += 1
        print(self.a)
        time.sleep(1)


class Test:
    def __init__(self):
        self.pool = Pool(5)
        self.ob = A()

    def spawn(self):
        return self.pool.apply_async(self.ob.foo, (), error_callback=err)

    def close(self):
        self.pool.close()


if __name__ == "__main__":
    print("hello")
    t = Test()

    for _ in range(10):
        t.spawn().get()
    t.close()
