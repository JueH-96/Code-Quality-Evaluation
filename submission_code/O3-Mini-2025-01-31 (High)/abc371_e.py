def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    # Read the list of integers
    arr = list(map(int, data[1:1+n]))
    
    # Total number of subarrays is T = n*(n+1)/2 (using integer division)
    T = n * (n + 1) // 2

    # For each distinct value, we want to count the number of subarrays in which it appears.
    # Notice that for a subarray, f(l, r) is the number of distinct values in it.
    # Hence, if we sum f(l, r) for all subarrays it equals:
    #    Sum_v (number of subarrays that contain v)
    #
    # How to compute “number of subarrays that contain v” if v appears in positions p1, p2, ... pk?
    # The key is to note that any subarray that does NOT contain v must lie entirely in one of the gap segments:
    #  1) before the first occurrence: positions [1, p1-1]
    #  2) between two consecutive occurrences: positions [p_i+1, p_{i+1}-1]
    #  3) after the last occurrence: positions [p_k+1, n]
    # The number of subarrays fully contained in a segment of length L is L*(L+1)//2.
    # Then, for a fixed v, the number of subarrays that contain v = T - (sum of subarrays in each gap).
    
    from collections import defaultdict
    positions_dict = defaultdict(list)
    # Use 1-indexing for positions.
    for idx, value in enumerate(arr, start=1):
        positions_dict[value].append(idx)
    
    answer = 0
    # Process each distinct value
    for value, positions in positions_dict.items():
        gap_total = 0
        # Gap before the first occurrence
        first_gap = positions[0] - 1
        if first_gap > 0:
            gap_total += first_gap * (first_gap + 1) // 2
        
        # Gaps between consecutive occurrences
        for i in range(len(positions) - 1):
            gap = positions[i+1] - positions[i] - 1
            if gap > 0:
                gap_total += gap * (gap + 1) // 2
        
        # Gap after the last occurrence
        last_gap = n - positions[-1]
        if last_gap > 0:
            gap_total += last_gap * (last_gap + 1) // 2
        
        # Subarrays that contain this value
        subarrays_with_value = T - gap_total
        answer += subarrays_with_value
    
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()