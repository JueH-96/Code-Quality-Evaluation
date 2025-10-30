n, L, R = map(int, input().split())
A = list(map(int, input().split()))
result = [str(max(L, min(R, a))) for a in A]
print(' '.join(result))