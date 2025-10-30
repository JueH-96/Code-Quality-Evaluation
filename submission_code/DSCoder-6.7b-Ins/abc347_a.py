# YOUR CODE HERE

import sys

N, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

multiples = [a for a in A if a % K == 0]
quotients = [a // K for a in multiples]

print(' '.join(map(str, sorted(quotients))))