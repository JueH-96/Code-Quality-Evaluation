def main():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    pairs = []
    idx = 2
    sum_b = 0
    for _ in range(N):
        a = int(input_data[idx]); b = int(input_data[idx+1])
        idx += 2
        pairs.append((a, b))
        sum_b += b

    # Sort by a_i
    pairs.sort(key=lambda x: x[0])
    A = [p[0] for p in pairs]
    B = [p[1] for p in pairs]
    
    # Prefix sum of B for fast range-sum queries
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + B[i]
    
    max_a = A[-1]
    # Binary search for the first day on or after day=1 such that total pills <= K
    # The day range is [1, max_a+1]
    low, high = 1, max_a+1
    
    while low < high:
        mid = (low + high) // 2
        # We want to calculate how many b_i are counted if a_i >= mid
        # index = number of a_i < mid
        # sum_below = prefix[index], so total pills for day mid is sum_b - sum_below
        index = bisect.bisect_left(A, mid)
        pills = sum_b - prefix[index]
        
        if pills <= K:
            high = mid
        else:
            low = mid + 1
    
    print(low)

# Call main() at the end
if __name__ == "__main__":
    main()