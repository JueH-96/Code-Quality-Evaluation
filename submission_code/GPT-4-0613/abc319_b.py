N = int(input().strip())
s = ['-'] * (N+1)
for j in range(1, 10):
    if N % j == 0:
        for i in range(0, N+1, N//j):
            if s[i] == '-':
                s[i] = str(j)
print(''.join(s))