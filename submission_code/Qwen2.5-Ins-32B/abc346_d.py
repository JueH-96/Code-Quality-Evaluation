import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:]))
    
    # Calculate the cost of flipping each bit
    flip_cost = [C[i] if S[i] == '0' else C[i] for i in range(N)]
    
    # Calculate the minimum cost to make the string good
    min_cost = float('inf')
    for i in range(1, N):
        # Calculate the cost of making the i-th and (i+1)-th characters the same
        cost = sum(flip_cost[:i]) + sum(flip_cost[i+1:])
        if S[i] == S[i-1]:
            min_cost = min(min_cost, cost)
        else:
            min_cost = min(min_cost, cost - min(flip_cost[i-1], flip_cost[i]))
    
    print(min_cost)

solve()