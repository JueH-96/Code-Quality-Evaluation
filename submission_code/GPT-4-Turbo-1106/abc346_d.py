def find_min_cost(N, S, C):
    min_cost = float('inf')
    for i in range(N - 1):
        if S[i] == S[i + 1]:
            continue
        cost = 0
        for j in range(N):
            if j == i or j == i + 1:
                continue
            if S[j] == S[j + 1]:
                cost += C[j]
        min_cost = min(min_cost, cost)
    return min_cost

# Read input from stdin
N = int(input().strip())
S = input().strip()
C = list(map(int, input().strip().split()))

# Find the minimum cost and print it to stdout
print(find_min_cost(N, S, C))