# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
takahashi_score = 0
aoki_score = 0

for i in range(1, 2 * N, 2):
    takahashi_score += int(input[i])
    aoki_score += int(input[i + 1])

if takahashi_score > aoki_score:
    print("Takahashi")
elif aoki_score > takahashi_score:
    print("Aoki")
else:
    print("Draw")