# YOUR CODE HERE
import sys

S = sys.stdin.read().strip()

result = "Yes"
for i in range(1, 16, 2):
    if S[i] != '0':
        result = "No"
        break

print(result)