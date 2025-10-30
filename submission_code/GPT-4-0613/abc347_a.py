N, K = map(int, input().split())
A = list(map(int, input().split()))

quotients = [a//K for a in A if a % K == 0]
print(' '.join(map(str, quotients)))