# YOUR CODE HERE
N = int(input())
a = N // 100  # hundreds digit
b = (N // 10) % 10  # tens digit
c = N % 10  # ones digit

first_int = b * 100 + c * 10 + a
second_int = c * 100 + a * 10 + b

print(f"{first_int} {second_int}")