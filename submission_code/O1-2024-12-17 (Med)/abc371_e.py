# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    from collections import defaultdict
    positions = defaultdict(list)
    
    # Record the (1-based) positions of each element
    for i, val in enumerate(A):
        positions[val].append(i + 1)

    # Total number of sub-subarrays
    T = N * (N + 1) // 2

    answer = 0
    
    # For each distinct value, compute how many sub-subarrays do NOT contain it
    # and subtract that from T to get how many sub-subarrays DO contain it
    for val, pos_list in positions.items():
        sum_of_gaps = 0
        prev = 0
        
        for p in pos_list:
            gap_len = p - prev - 1
            sum_of_gaps += gap_len * (gap_len + 1) // 2
            prev = p
        
        # Handle gap after the last occurrence
        gap_len = (N + 1) - prev - 1
        sum_of_gaps += gap_len * (gap_len + 1) // 2

        # Number of sub-subarrays containing this value
        contain_val = T - sum_of_gaps
        answer += contain_val

    print(answer)

# Don't forget to call main()!
main()