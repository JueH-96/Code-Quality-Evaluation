N, K = map(int, input().split())
A = list(map(int, input().split()))
quotients = [x // K for x in A if x % K == 0]
quotients.sort()
print(' '.join(map(str, quotients)))