# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
A = list(map(int, input[2:]))

quotients = [a // K for a in A if a % K == 0]
quotients.sort()

print(" ".join(map(str, quotients)))