import sys

def main():
    input = sys.stdin.readline
    N, D, P = map(int, input().split())
    F = list(map(int, input().split()))
    
    # Total sum if we never buy any passes
    total_fare = sum(F)
    
    # Sort fares in descending order so that we can apply passes to the most expensive days
    F.sort(reverse=True)
    
    # Build prefix sums of the sorted fares
    # prefix[i] = sum of the top i fares
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + F[i]
    
    # Maximum number of batches we might consider buying
    # (once K*D >= N, buying more won't cover extra days)
    max_batches = (N + D - 1) // D
    
    # Start with the cost of buying 0 batches (just pay all regular fares)
    ans = total_fare
    
    # Try buying k batches, for k = 1..max_batches
    for k in range(1, max_batches + 1):
        covered_days = k * D
        if covered_days > N:
            covered_days = N
        # Cost = cost of k batches + pay full fare for the uncovered days
        cost = k * P + (total_fare - prefix[covered_days])
        if cost < ans:
            ans = cost
    
    print(ans)

# Call the main function
main()