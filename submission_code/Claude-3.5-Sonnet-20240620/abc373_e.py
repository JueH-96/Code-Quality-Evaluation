# YOUR CODE HERE
def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    remaining_votes = K - sum(A)
    sorted_votes = sorted([(votes, i) for i, votes in enumerate(A)], reverse=True)
    
    result = [-1] * N
    for i in range(N):
        if i < M:
            target = sorted_votes[M-1][0]
            needed = max(0, target + 1 - A[i])
            if needed <= remaining_votes:
                result[i] = needed
        elif A[i] > sorted_votes[M-1][0]:
            result[i] = 0
    
    print(*result)

solve()