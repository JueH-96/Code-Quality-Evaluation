# YOUR CODE HERE
import sys

def solve():
    """
    Reads an interval [L, R) and decomposes it into a minimum number of
    "good" intervals, printing the count and the intervals themselves.
    A "good" interval [l, r) is one where r-l is a power of 2 and l is a
    multiple of r-l.
    """
    
    # Read L and R from a single line of input.
    # Using sys.stdin.readline for potentially faster I/O with large inputs.
    try:
        L, R = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # This case is unlikely in a controlled environment like an online judge.
        return

    # current_l tracks the start of the remaining interval [current_l, R)
    # that we need to partition.
    current_l = L
    
    # The result list will store the pairs (l_k, r_k) of the partition.
    result = []
    
    # We continue as long as our current position is less than the right boundary R.
    while current_l < R:
        # The greedy strategy is to find the largest valid "good" interval starting
        # at current_l. The step size `step = r - l` must be a power of 2, say 2^k.
        
        if current_l == 0:
            # Special case: 0 is a multiple of any power of 2.
            # We just need to find the largest k such that 2^k <= R.
            # Since current_l < R, R must be positive.
            # k = floor(log2(R)), which is R.bit_length() - 1.
            k = R.bit_length() - 1
            step = 1 << k
        else:
            # For current_l > 0, we need the largest k such that 2^k divides
            # current_l and 2^k <= R - current_l.
            # This means k <= v_2(current_l) and k <= floor(log2(R - current_l)).
            # We want the maximum such k.
            
            # v_2(current_l) is the number of trailing zeros in the binary representation
            # of current_l. (x & -x) isolates the lowest set bit.
            # .bit_length() - 1 gives the exponent of this power of 2.
            v2_current_l = (current_l & -current_l).bit_length() - 1
            
            # floor(log2(R - current_l))
            diff = R - current_l
            floor_log2_diff = diff.bit_length() - 1
            
            # The optimal k is the minimum of these two constraints.
            k = min(v2_current_l, floor_log2_diff)
            step = 1 << k

        # Append the found interval to the results.
        result.append((current_l, current_l + step))
        
        # Update our position to the end of the interval we just added.
        current_l += step
        
    # Print the total number of intervals in the minimal partition.
    print(len(result))
    
    # Print each interval pair on its own line.
    for l, r in result:
        print(l, r)

solve()