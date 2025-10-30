def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # We need to find at least two distinct subsequences of A that match B
    # Let's use a two-pointer technique to count how many times we can find B in A
    
    # First, find all positions in A for each element in B
    from collections import defaultdict
    positions = defaultdict(list)
    for i, a in enumerate(A):
        positions[a].append(i)
    
    # Now, try to match B in A using these positions
    # We need to find B such that indices are strictly increasing
    import bisect
    
    def count_ways_to_form_B():
        # We will use a dynamic list of indices where we can place the next B[j]
        current_indices = [-1]  # Start with an imaginary position before the start of A
        for b in B:
            if b not in positions:
                return 0  # If any B[j] cannot be found in A, we can't form B
            
            # For each b in B, we need to find valid positions in A that are greater than the last used index
            next_indices = []
            for pos in positions[b]:
                # Use binary search to find the first position that is greater than the last used index in current_indices
                idx = bisect.bisect_right(current_indices, pos)
                if idx > len(next_indices):
                    next_indices.append(pos)
                elif idx == len(next_indices):
                    next_indices.append(pos)
                else:
                    next_indices[idx] = min(next_indices[idx], pos)
            
            if not next_indices:
                return 0  # If we can't find any valid positions for the current b, no subsequence can be formed
            
            current_indices = next_indices
        
        return len(current_indices)
    
    # We need at least two ways to form B
    if count_ways_to_form_B() >= 2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()