# YOUR CODE HERE
def count_vertices_at_distance(N, X, K):
    if K == 0:
        return 1 if X <= N else 0
    
    lower = X * (2 ** K)
    upper = (X + 1) * (2 ** K) - 1
    
    if lower > N:
        return 0
    
    count = min(upper, N) - lower + 1
    
    if K > 1:
        parent = X // 2
        if parent > 0:
            count += count_vertices_at_distance(N, parent, K - 1)
    
    return count

T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(count_vertices_at_distance(N, X, K))