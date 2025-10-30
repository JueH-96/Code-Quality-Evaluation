# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))
multiples = [a for a in A if a % K == 0]
quotients = [a // K for a in multiples]
quotients.sort()
print(' '.join(map(str, quotients)))