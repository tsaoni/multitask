class MyClass:
    my_static_var = 42
    
    def my_method(self, my_param=my_static_var):
        print(my_param)

if __name__ == '__main__':
    c = MyClass()
    c.my_method()