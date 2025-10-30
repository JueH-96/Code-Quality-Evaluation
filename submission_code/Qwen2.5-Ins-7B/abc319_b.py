# YOUR CODE HERE
n = int(input())
s = ['-'] * (n + 1)
for i in range(1, 10):
    for j in range(1, n + 1):
        if n % (i * j) == 0:
            s[n // (i * j) - 1] = str(i)
print(''.join(s))