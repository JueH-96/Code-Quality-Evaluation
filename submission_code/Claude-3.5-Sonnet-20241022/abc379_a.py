# Read the three-digit number from input
N = int(input())

# Extract the digits
a = N // 100        # hundreds digit
b = (N // 10) % 10  # tens digit
c = N % 10          # ones digit

# Form the two required numbers
# First number: bca (b*100 + c*10 + a)
num1 = b * 100 + c * 10 + a

# Second number: cab (c*100 + a*10 + b)
num2 = c * 100 + a * 10 + b

# Print the result
print(f"{num1} {num2}")