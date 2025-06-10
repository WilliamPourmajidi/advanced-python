def add_two_numbers (a,b):
    # business logic
    return a + b

c = 10
d = 20

f = add_two_numbers(c,d)
print("coming from function", f)


x = lambda a, b : a + b
print("coming from lambda function",x(10, 20))

w = lambda x,y,z : x + y + z
print(w(2,3,4))

def add(a,b,c):
    """
    This function takes 3 values and add them together
    :param a: integer
    :param b: integer
    :param c: integer
    :return: integer
    """
    sum = a + b + c
    return sum

print (add)


########################
import numpy as np
matrix=np.array([[1,2,3], [2,4,6], [7, 8,9]])
print(matrix)

add_100 = lambda i: i + 100
# #Challenge question : What does lambda function do?
vectrorized_100 = np.vectorize(add_100)
# #Challenge question : explore np.vectorize
print(add_100)
print(vectrorized_100(matrix))