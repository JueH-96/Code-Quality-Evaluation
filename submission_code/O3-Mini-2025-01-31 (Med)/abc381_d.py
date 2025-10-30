def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    if N < 2:
        # if there is less than 2 elements then no contiguous subarray can be non-empty
        # because valid 1122 subarray must have even length.
        sys.stdout.write("0")
        return

    A = list(map(int, data[1:]))
    
    # The idea:
    # A valid 1122 subarray, when re-indexed, must have even length.
    # Partition it as (x, x), (y, y), (z, z) ... and by condition each integer appears exactly twice,
    # so the values for each pair must be distinct.
    #
    # Suppose we choose a contiguous subarray from A with indices i...j.
    # For it to be a 1122 sequence it must be that:
    #   1. Its length L = j-i+1 is even; let m = L/2.
    #   2. For each k from 0 to m-1, we require A[i+2*k] == A[i+2*k+1].
    #   3. All these pair values (A[i], A[i+2], ..., A[i+2*(m-1)]) are distinct.
    #
    # Notice that once a subarray starting at i meets condition (2),
    # the pairs come from positions i, i+2, i+4, ... 
    # and the positions come with a fixed parity relative to i.
    #
    # So for each possible starting parity p in {0, 1} (0-indexed),
    # we look at positions i = p, p+2, p+4, ... where a valid "pair start" occurs,
    # i.e. A[i]==A[i+1]. They form a contiguous block in A,
    # and the largest valid subarray from that block will be the longest segment,
    # whose underlying pair values are all distinct.
    #
    # We can scan A (by stepping in increments of 2) for each parity p.
    # When we find a valid pair, we accumulate a block (list of pair values)
    # as long as the pairs remain contiguous (i.e. at indices i, i+2, etc.).
    # Then within that block, we use a two-pointer sliding window to find the maximum
    # contiguous segment that contains distinct values. Its length (in pairs)
    # multiplied by 2 is the length of a 1122 sequence.
    
    ans = 0
    # Check both parities: because if a valid subarray starts at i, then i can be either even or odd.
    for p in (0, 1):
        i = p
        while i < N - 1:
            if A[i] == A[i+1]:
                block = []
                # Build a contiguous block of valid pairs.
                # The pairs we consider are at positions i, i+2, i+4, ... 
                # They are "contiguous" if each pair at A[i] and A[i+1] satisfies A[i]==A[i+1].
                while i < N - 1 and A[i] == A[i+1]:
                    block.append(A[i])
                    i += 2
                # Now, 'block' holds the values for pairs in one contiguous valid segment of pairs.
                # We now want the longest contiguous segment of this block that has all distinct values.
                # This corresponds to a valid 1122 subarray which uses that many consecutive pairs.
                l = 0
                cur_set = set()
                cur_max = 0
                for r, val in enumerate(block):
                    # Slide the left pointer until we remove duplicate val.
                    while val in cur_set:
                        cur_set.remove(block[l])
                        l += 1
                    cur_set.add(val)
                    cur_max = max(cur_max, r - l + 1)
                candidate_length = cur_max * 2   # each valid pair contributes 2 numbers.
                ans = max(ans, candidate_length)
            else:
                # If the current position does not start a valid pair, just move on.
                i += 2

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()