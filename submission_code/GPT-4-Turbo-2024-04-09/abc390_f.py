def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # We need to calculate f(L, R) for all 1 <= L <= R <= N
    # To do this efficiently, we will use a divide and conquer approach similar to the one used in the "Minimum number of removals to make mountain array"
    
    from collections import defaultdict
    
    def min_operations(L, R):
        if L > R:
            return 0
        if L == R:
            return 1
        
        # Find the most frequent element in A[L:R+1]
        freq = defaultdict(int)
        max_freq = 0
        most_frequent = None
        
        for i in range(L, R+1):
            freq[A[i]] += 1
            if freq[A[i]] > max_freq:
                max_freq = freq[A[i]]
                most_frequent = A[i]
        
        # Positions of the most frequent element
        positions = [i for i in range(L, R+1) if A[i] == most_frequent]
        
        # Calculate the minimum operations needed
        # Start by considering the whole segment is erased by covering the most frequent element
        min_ops = 1
        
        # Now, we need to handle the segments between the positions of the most frequent element
        current_min_ops = min_ops
        last_pos = L - 1
        for pos in positions:
            # Solve for the segment before this position of the most frequent element
            current_min_ops += min_operations(last_pos + 1, pos - 1)
            last_pos = pos
        
        # There might be a tail segment after the last occurrence of the most frequent element
        current_min_ops += min_operations(last_pos + 1, R)
        
        return current_min_ops
    
    # We need to sum up f(L, R) for all 1 <= L <= R <= N
    total_sum = 0
    for L in range(1, N+1):
        for R in range(L, N+1):
            total_sum += min_operations(L-1, R-1)  # Convert to 0-based index
    
    print(total_sum)