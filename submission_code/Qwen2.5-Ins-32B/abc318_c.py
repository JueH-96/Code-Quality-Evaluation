import sys

def solve():
    N, D, P = map(int, input().split())
    F = list(map(int, input().split()))
    
    # Calculate the total cost if no passes are used
    total_cost = sum(F)
    
    # Sort the fares in descending order to maximize the benefit of passes
    F.sort(reverse=True)
    
    # Calculate the cost reduction by using passes
    cost_reduction = 0
    for i in range(0, N, D):
        cost_reduction += min(P, sum(F[i:i+D]))
    
    # The minimum cost is the total cost minus the cost reduction
    print(total_cost - cost_reduction)

if __name__ == "__main__":
    solve()