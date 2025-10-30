# YOUR CODE HERE
import sys
input = sys.stdin.read
S = input().strip()
count = 0
i = 0
while i < len(S):
    if i + 1 < len(S) and S[i] == '1' and S[i + 1] == '0':
        count += 1
        i += 2
    else:
        count += 1
        i += 1
print(count)