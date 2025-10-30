import sys
N = int(sys.stdin.read().strip())
s = ['-'] * N
if N % 2 == 1:  # odd
    mid = N // 2
    s[mid] = '='
else:  # even
    idx1 = N // 2 - 1
    idx2 = N // 2
    s[idx1] = '='
    s[idx2] = '='
result = ''.join(s)
print(result)