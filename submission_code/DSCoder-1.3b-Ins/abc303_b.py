# YOUR CODE HERE

N, M = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(M)]

def count_bad_pairs(N, M, a):
    bad_pairs = 0
    for i in range(M):
        for j in range(1, N):
            if a[i][j] == a[i][j-1] + 1 or a[i][j] == a[i][j-1] - 1:
                bad_pairs += 1
    return bad_pairs

print(count_bad_pairs(N, M, a))