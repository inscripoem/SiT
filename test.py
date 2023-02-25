

def decorator(func):
    def wrapper(*args, **kwargs):
        print('before')
        func(*args, **kwargs)
        print('after')
    return wrapper

@decorator
def test():
    print('test')

if  __name__ == "__main__":
    test()