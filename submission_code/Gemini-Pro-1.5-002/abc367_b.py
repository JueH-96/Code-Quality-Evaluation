# YOUR CODE HERE
x = input()
if '.' in x:
    x = x.rstrip('0')
    if x[-1] == '.':
        x = x[:-1]
print(x)