def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    A = list(map(int, data[1:]))

    # A 1122 sequence is an even‐length subarray which can be partitioned into
    # pairs of successive identical numbers, and every number appears exactly in one pair.
    # In other words, if a valid contiguous subarray starts at index L and has length 2*k,
    # then we must have:
    #   • A[L] == A[L+1], A[L+2] == A[L+3], ..., A[L+2*k-2] == A[L+2*k-1]
    #   • The values A[L], A[L+2], ..., A[L+2*k-2] are all distinct.
    #
    # Note: The subarray can start at either an even or an odd index.
    # So we solve the problem by considering both parities separately.
    #
    # Our plan:
    #  1. For each parity p in {0,1}, consider indices i with that parity where i+1 exists.
    #  2. If A[i] == A[i+1] we have a valid pair. However, for a valid 1122 subarray,
    #     the pairs must appear contiguously – that is, if we use a pair at index i, then
    #     the next pair must be at index i+2.
    #  3. We will scan the array (only indices of the given parity) and group consecutive
    #     valid pairs into segments. In each segment, we have a list of pair values.
    #  4. In any 1122 subarray, the list of pair values (one from each pair) must be distinct,
    #     so within each segment we compute the longest contiguous subsegment with distinct values.
    #  5. The subarray length in A will then be 2 times (number of pairs in the subsegment).
    
    global_max = 0
    
    for p in (0, 1):
        segments = []  # Each segment is a contiguous chain (by original indices) of valid pair blocks.
        curr_seg = []  # List to hold the pair values (each pair is represented by the repeated value)
        prev_index = None  # To ensure the pairs occur contiguously, i must equal the previous index + 2.
        
        # We iterate over indices of parity p (ensuring i+1 exists).
        for i in range(p, N - 1, 2):
            if A[i] == A[i+1]:
                # A valid pair is found.
                if not curr_seg:
                    curr_seg = [A[i]]
                    prev_index = i
                else:
                    if i == prev_index + 2:
                        curr_seg.append(A[i])
                        prev_index = i
                    else:
                        # There is a gap: the contiguous chain is broken.
                        segments.append(curr_seg)
                        curr_seg = [A[i]]
                        prev_index = i
            else:
                # When we hit an index where the pair is not equal, finish current chain (if any).
                if curr_seg:
                    segments.append(curr_seg)
                    curr_seg = []
                prev_index = None
        if curr_seg:
            segments.append(curr_seg)
        
        # For each segment, we want the longest contiguous subsegment (i.e. a sequence of consecutive pairs)
        # such that all pair values are distinct.
        # We use a two‐pointer (sliding window) technique.
        for seg in segments:
            start = 0
            freq = {}
            maxi = 0
            for end, val in enumerate(seg):
                freq[val] = freq.get(val, 0) + 1
                # If a duplicate appears, slide the window from the left until it is distinct.
                while freq[val] > 1:
                    freq[seg[start]] -= 1
                    start += 1
                curr_window = end - start + 1
                if curr_window > maxi:
                    maxi = curr_window
            candidate_length = maxi * 2  # Each pair contributes two elements.
            if candidate_length > global_max:
                global_max = candidate_length

    sys.stdout.write(str(global_max))

if __name__ == '__main__':
    main()