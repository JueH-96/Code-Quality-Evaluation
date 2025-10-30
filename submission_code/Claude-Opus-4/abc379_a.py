# YOUR CODE HERE
N = int(input())

# Extract digits
a = N // 100        # hundreds digit
b = (N // 10) % 10  # tens digit
c = N % 10          # ones digit

# Form new numbers
first = b * 100 + c * 10 + a
second = c * 100 + a * 10 + b

print(first, second)