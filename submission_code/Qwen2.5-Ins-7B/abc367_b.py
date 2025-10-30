# YOUR CODE HERE
x = float(input().strip())
x_rounded = round(x, 3)
x_str = str(x_rounded)
if x_str[-1] == '0' and x_str[-2] == '.':
    x_str = x_str[:-2]
print(x_str)