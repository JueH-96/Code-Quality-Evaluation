import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N, Q = int(data[0]), int(data[1])
    S = data[2]
    queries = [(int(data[i]), int(data[i+1])) for i in range(3, 2*Q+2, 2)]
    
    # Precompute prefix sums for pairs of consecutive equal characters
    prefix_sums = [0] * (N + 1)
    for i in range(1, N):
        if S[i] == S[i-1]:
            prefix_sums[i+1] = prefix_sums[i] + 1
        else:
            prefix_sums[i+1] = prefix_sums[i]
    
    # Answer each query using the prefix sums
    for l, r in queries:
        print(prefix_sums[r] - prefix_sums[l])

solve()