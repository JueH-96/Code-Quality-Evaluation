# YOUR CODE HERE
n = int(input())
a = n // 100
b = (n % 100) // 10
c = n % 10

first_integer = b * 100 + c * 10 + a
second_integer = c * 100 + a * 10 + b

print(first_integer, second_integer)