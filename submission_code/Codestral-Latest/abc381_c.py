# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

def is_11_22_string(s):
    length = len(s)
    if length % 2 == 0:
        return False
    mid = length // 2
    if s[mid] != '/':
        return False
    for i in range(mid):
        if s[i] != '1':
            return False
    for i in range(mid + 1, length):
        if s[i] != '2':
            return False
    return True

max_length = 0
for i in range(N):
    for j in range(i + 1, N + 1):
        if is_11_22_string(S[i:j]):
            max_length = max(max_length, j - i)

print(max_length)