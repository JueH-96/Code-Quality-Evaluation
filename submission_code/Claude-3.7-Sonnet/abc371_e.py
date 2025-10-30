def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    total_sum = 0
    
    # For each starting position i
    for i in range(N):
        # Keep track of distinct elements
        distinct_elements = set()
        
        # For each ending position j (where j ≥ i)
        for j in range(i, N):
            # Add the current element to our set
            distinct_elements.add(A[j])
            
            # Add the count of distinct elements to our total
            total_sum += len(distinct_elements)
    
    return total_sum

print(solve())