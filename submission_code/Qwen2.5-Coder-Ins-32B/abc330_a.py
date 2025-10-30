# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
L = int(input[1])
scores = list(map(int, input[2:]))

passed_count = sum(1 for score in scores if score >= L)

print(passed_count)