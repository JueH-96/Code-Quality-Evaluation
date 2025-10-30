# YOUR CODE HERE
def is_valid_fixed_count(N, K):
    if K == 0 or K == N*N:
        return True
    if K == 1 or K == N*N - 1:
        return False
    if K <= N or K >= N*N - N:
        return True
    return False

N, Q = map(int, input().split())

for _ in range(Q):
    K = int(input())
    print("Yes" if is_valid_fixed_count(N, K) else "No")