def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    L = int(data[0])
    R = int(data[1])
    
    intervals = []
    current = L
    # While there is still a part of S(L, R) uncovered.
    while current < R:
        # The maximum allowed block size is determined by the remaining length.
        rem = R - current
        # Starting with the largest possible power-of-2 block that can possibly fit.
        # floor_log2 returns the largest exponent k such that 2^k <= rem.
        if rem > 0:
            k = rem.bit_length() - 1  # since 2^k <= rem < 2^(k+1)
        else:
            k = 0
        
        # Among blocks of size 2^k, 2^(k-1), ... , 1, choose the largest one such that
        # the block starts at a multiple of its size.
        chosen_exponent = None
        for exp in range(k, -1, -1):
            block_size = 1 << exp  # 2^exp
            # Check "good" condition: the starting number must be a multiple of block_size.
            if current % block_size == 0:
                chosen_exponent = exp
                break
        if chosen_exponent is None:
            # This fallback should never happen because exp=0 always works.
            chosen_exponent = 0
        block_size = 1 << chosen_exponent
        
        # It's possible that the chosen block size is too big for the remaining segment.
        # But our loop on exponent ensured block_size <= rem.
        l_val = current
        r_val = current + block_size
        intervals.append((l_val, r_val))
        current = r_val
        
    # Print output: first the number of intervals, then the intervals in order.
    out_lines = [str(len(intervals))]
    for l_val, r_val in intervals:
        out_lines.append(f"{l_val} {r_val}")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()