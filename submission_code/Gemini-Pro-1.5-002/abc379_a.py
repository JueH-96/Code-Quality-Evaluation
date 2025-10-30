# YOUR CODE HERE
n = int(input())
a = n // 100
b = (n % 100) // 10
c = n % 10

num1 = b * 100 + c * 10 + a
num2 = c * 100 + a * 10 + b

print(num1, num2)