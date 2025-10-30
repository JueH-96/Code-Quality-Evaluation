def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Double the array to handle circular nature
    A_double = A + A
    
    def can_divide(min_weight):
        # Try each starting position
        for start in range(N):
            segments = 0
            i = start
            
            while segments < K and i < start + N:
                current_sum = 0
                segment_start = i
                
                # Accumulate until we reach min_weight
                while current_sum < min_weight and i < start + N:
                    current_sum += A_double[i]
                    i += 1
                
                if current_sum >= min_weight:
                    segments += 1
                else:
                    break
            
            if segments == K and i == start + N:
                return True
        
        return False
    
    # Binary search for maximum minimum weight
    left, right = 1, sum(A) // K
    
    while left < right:
        mid = (left + right + 1) // 2
        if can_divide(mid):
            left = mid
        else:
            right = mid - 1
    
    max_min_weight = left
    
    # Find all optimal divisions
    def get_optimal_divisions():
        divisions = []
        
        for start in range(N):
            segments = []
            i = start
            
            while len(segments) < K and i < start + N:
                current_sum = 0
                segment_start = i
                
                while current_sum < max_min_weight and i < start + N:
                    current_sum += A_double[i]
                    i += 1
                
                if current_sum >= max_min_weight:
                    segments.append((segment_start, i - 1))
                else:
                    break
            
            if len(segments) == K and i == start + N:
                divisions.append(segments)
        
        return divisions
    
    divisions = get_optimal_divisions()
    
    # Count unused cut lines
    # Cut line i is between piece i and piece (i+1)%N
    used_cuts = set()
    
    for division in divisions:
        for segment_start, segment_end in division:
            # The cut after segment_end is used (except for the last segment)
            if segment_end < segment_start + N - 1:  # Not the last piece of the division
                used_cuts.add(segment_end % N)
    
    unused_cuts = N - len(used_cuts)
    
    print(max_min_weight, unused_cuts)

solve()