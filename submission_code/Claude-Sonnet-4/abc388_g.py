def solve_query(mochi):
    n = len(mochi)
    if n < 2:
        return 0
    
    count = 0
    i = 0  # pointer for smaller mochi (top)
    j = 0  # pointer for larger mochi (bottom)
    
    used = [False] * n
    
    while i < n:
        if used[i]:
            i += 1
            continue
            
        # Find the first unused mochi that can serve as bottom
        j = max(j, i + 1)
        found = False
        
        while j < n:
            if not used[j] and mochi[i] * 2 <= mochi[j]:
                used[i] = True
                used[j] = True
                count += 1
                found = True
                j += 1
                break
            j += 1
        
        if not found:
            i += 1
    
    return count

# Read input
n = int(input())
A = list(map(int, input().split()))
q = int(input())

for _ in range(q):
    l, r = map(int, input().split())
    # Convert to 0-based indexing
    mochi_subset = A[l-1:r]
    result = solve_query(mochi_subset)
    print(result)