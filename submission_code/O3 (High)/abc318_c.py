import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, D, P = data[0], data[1], data[2]
    fares = data[3:]
    
    # Sort fares in descending order so that the most expensive days
    # can be covered by passes first.
    fares.sort(reverse=True)
    
    # Prefix sums: prefix[i] = sum of the i most expensive fares
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + fares[i - 1]
    
    total_fares = prefix[N]
    
    # We need to check k = 0 .. ceil(N / D)
    max_batches = (N + D - 1) // D
    
    best_cost = total_fares          # k = 0 (buy no passes)
    
    for k in range(1, max_batches + 1):
        passes_bought = k * D
        days_covered = min(passes_bought, N)
        cost = k * P + (total_fares - prefix[days_covered])
        if cost < best_cost:
            best_cost = cost
    
    print(best_cost)

if __name__ == "__main__":
    main()