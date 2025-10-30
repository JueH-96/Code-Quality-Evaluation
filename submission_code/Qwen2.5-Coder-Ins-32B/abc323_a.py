# YOUR CODE HERE
import sys

S = sys.stdin.read().strip()

for i in range(1, 8):  # Even indices from 2 to 16 correspond to indices 1 to 7 in 0-based index
    if S[2 * i] != '0':
        print("No")
        break
else:
    print("Yes")