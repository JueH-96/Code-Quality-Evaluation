# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
A = list(map(int, data[:9]))
B = list(map(int, data[9:]))

score_takahashi = sum(A)
score_aoki = sum(B)

runs_needed = (score_takahashi - score_aoki) + 1

print(runs_needed)