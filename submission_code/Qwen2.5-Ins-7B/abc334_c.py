# N, K = map(int, input().split())
# A = list(map(int, input().split()))

def min_weirdness(N, K, A):
    A = [0] + A + [N+1]
    A.sort()
    pairs = (2*N - K) // 2
    total_weirdness = 0
    for i in range(1, len(A) - 1):
        total_weirdness += A[i+1] - A[i] - 1
    return total_weirdness - (pairs - K) * 1

N, K = map(int, input().split())
A = list(map(int, input().split()))
print(min_weirdness(N, K, A))