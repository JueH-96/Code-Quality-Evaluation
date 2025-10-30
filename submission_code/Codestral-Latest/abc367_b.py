# YOUR CODE HERE
x = input().strip()
# Remove trailing zeros after the decimal point
x = x.rstrip('0')
# If the last character is a decimal point, remove it
if x[-1] == '.':
    x = x[:-1]
print(x)