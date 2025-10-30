def max_kagamimochi(mochi, L, R):
    """
    Find the maximum number of kagamimochi we can form using mochi from L to R.
    """
    # Extract the mochi in the given range (adjust for 0-indexing)
    range_mochi = mochi[L-1:R]
    
    n = len(range_mochi)
    used = [False] * n
    count = 0
    
    # Start from the smallest mochi
    for i in range(n):
        if used[i]:
            continue
        
        # Try to find the smallest eligible partner
        for j in range(i+1, n):
            if used[j]:
                continue
            
            if range_mochi[i] * 2 <= range_mochi[j]:
                used[i] = used[j] = True
                count += 1
                break
    
    return count

# Read the input
N = int(input())
mochi = list(map(int, input().split()))
Q = int(input())

# Process each query and print the result
for _ in range(Q):
    L, R = map(int, input().split())
    print(max_kagamimochi(mochi, L, R))