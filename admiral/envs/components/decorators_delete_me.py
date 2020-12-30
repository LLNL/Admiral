
# def my_decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

# def say_whee():
#     print("Whee!")

# say_whee = my_decorator(say_whee)
# # say_whee is the same as the wrapper that is returned above in my_decorator
# # it has a reference to the original say_whee function, and that reference is called
# # `func`.

# # This is the same as writing
# @my_decorator
# def say_whee():
#     print("Whee!")


# # Standard decorator syntax
# import functools

# def decorator(func):
#     @functools.wraps(func)
#     def inner(*args, **kwargs):
#         # Do something before
#         value = func(*args, **kwargs)
#         # Do something after
#         return value
#     return inner



# Deactorator that takes in arguments like so:
# @repeat(num_times=4):
# def func():
#   return
#
# To do this, we need @repeat to return a function that can act as a decorator
# def repeat(num_times):
#   return decorator
#
# Then, this
# @repeat(num_times=4)
# def func():
#   return
# 
# is the same as this:
# @decorator
# def func():
#   return
def repeat(num_times): # repeat function that takes in argumetn and returns decorator
    def decorator(func): # decorator that causes the actual repeat to happen
        @functools.wraps(func) # maintain the information about the function
        def inner(*args, **kwargs): # The inner function
            for _ in range(num_times): # call the function 
                value = func(*args, **kwargs)
            return value
        return inner
    return decorator


# My example
from dataclasses import dataclass


def decorator(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return inner

@decorator
class Agent:
    id = None
    observation_space = None
    action_space = None

