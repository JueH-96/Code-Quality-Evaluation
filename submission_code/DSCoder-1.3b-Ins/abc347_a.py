# YOUR CODE HERE

N, K = map(int, input().split())
A = list(map(int, input().split()))

multiples = [i for i in A if i % K == 0]
quotients = [i / K for i in multiples]

print(*sorted(quotients))