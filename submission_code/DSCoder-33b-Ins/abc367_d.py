# YOUR CODE HERE

N, M = map(int, input().split())
A = list(map(int, input().split()))

def solve(N, M, A):
    total_steps = sum(A)
    pairs = 0
    for i in range(N):
        steps = 0
        for j in range(N):
            steps += A[(i+j)%N]
            if steps % M == 0:
                pairs += 1
    return pairs

print(solve(N, M, A))