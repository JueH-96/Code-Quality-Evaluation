# Read the three-digit integer N
N = int(input())

# Extract the hundreds (a), tens (b), and ones (c) digits
a = N // 100
b = (N // 10) % 10
c = N % 10

# Calculate the first number (b,c,a order)
first_num = b * 100 + c * 10 + a

# Calculate the second number (c,a,b order)
second_num = c * 100 + a * 10 + b

# Print the results
print(f"{first_num} {second_num}")