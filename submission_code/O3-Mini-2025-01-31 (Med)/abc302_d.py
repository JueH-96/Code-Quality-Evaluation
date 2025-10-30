def main():
    import sys, bisect

    data = sys.stdin.read().split()
    it = iter(data)
    
    N = int(next(it))
    M = int(next(it))
    D = int(next(it))
    
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # Sort the candidate lists
    A.sort()
    B.sort()
    
    best = -1
    # For each gift candidate for Aoki (with value a), find a candidate for Snuke (with value b)
    # such that b is in the range [a-D, a+D]. If such a b exists, a+b is a valid sum.
    # We use binary search to quickly find the range in B.
    for a in A:
        lower_bound = a - D  # b must be at least a-D
        upper_bound = a + D  # b must be at most a+D
        
        # Find the first index in B with value >= lower_bound.
        left_index = bisect.bisect_left(B, lower_bound)
        
        # Find the index after the last index in B with value <= upper_bound.
        right_index = bisect.bisect_right(B, upper_bound)
        
        # If there is any candidate in this range, choose the one with the maximum value (last element in this range)
        if left_index < right_index:
            candidate_b = B[right_index - 1]
            candidate_sum = a + candidate_b
            if candidate_sum > best:
                best = candidate_sum
    
    # Output the result. If no valid pair was found, best remains -1.
    sys.stdout.write(str(best))

# Ensure the main function is called.
if __name__ == '__main__':
    main()