# YOUR CODE HERE
x = float(input())
s = "{:.3f}".format(x)
s = s.rstrip('0')
if s.endswith('.'):
    s = s[:-1]
print(s)