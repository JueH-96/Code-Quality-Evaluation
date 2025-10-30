# YOUR CODE HERE
N = int(input())

# Extract digits
a = N // 100  # hundreds digit
b = (N // 10) % 10  # tens digit
c = N % 10  # ones digit

# Form the two required numbers
bca = b * 100 + c * 10 + a
cab = c * 100 + a * 10 + b

print(bca, cab)