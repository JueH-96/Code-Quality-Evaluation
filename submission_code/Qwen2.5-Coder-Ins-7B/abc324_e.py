# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = data[1]
S = data[2:]

def count_subsequence(s, t):
    t = list(t)
    for char in s:
        if not t:
            break
        if char == t[0]:
            t.pop(0)
    return len(t) == 0

count = 0
for i in range(N):
    for j in range(N):
        if count_subsequence(S[i] + S[j], T):
            count += 1

print(count)