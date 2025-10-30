# Read the input as an integer
N = int(input().strip())

# Extract the hundreds, tens, and ones digits
a = N // 100
b = (N // 10) % 10
c = N % 10

# Construct the two new numbers
num1 = b * 100 + c * 10 + a
num2 = c * 100 + a * 10 + b

# Output the results
print(num1, num2)