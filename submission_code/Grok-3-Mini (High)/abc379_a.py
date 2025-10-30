# Read the input integer N
N = int(input())

# Extract the hundreds, tens, and ones digits
a = N // 100  # Hundreds digit
b = (N // 10) % 10  # Tens digit
c = N % 10  # Ones digit

# Form the first new integer with digits b, c, a
new_num1 = b * 100 + c * 10 + a

# Form the second new integer with digits c, a, b
new_num2 = c * 100 + a * 10 + b

# Print the two integers separated by a space
print(new_num1, new_num2)