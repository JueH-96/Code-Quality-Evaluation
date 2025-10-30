def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()
    
    # First, identify all 1-blocks. Each block is a contiguous substring of '1's 
    # that is either at the beginning or immediately preceded by a '0',
    # and either at the end or immediately followed by a '0'.
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1':
            start = i
            while i + 1 < N and S[i + 1] == '1':
                i += 1
            end = i
            blocks.append((start, end))
        i += 1
    
    # It's guaranteed that there are at least K blocks.
    # Let the blocks be numbered 1..m in order. We want to move the K-th block
    # to immediately after the (K-1)-th block.
    prev_block = blocks[K - 2]  # (K-1)-th block in 1-index (0-index in our list)
    kth_block = blocks[K - 1]   # K-th block
    prev_r = prev_block[1]      # the last index of the (K-1)-th block (0-indexed)
    kth_l, kth_r = kth_block    # starting and ending indices (0-indexed) of the K-th block
    
    # According to the specification:
    # For positions 1-indexed:
    #   T_i = S_i for 1 <= i <= r_{K-1}
    #   T_i = 1 for i from r_{K-1}+1 to r_{K-1} + (r_K - l_K) + 1
    #   T_i = 0 for i from r_{K-1} + (r_K - l_K) + 2 to r_K
    #   T_i = S_i for i from r_K+1 to N
    # Converting these to 0-indexed:
    #   The unchanged prefix is S[0:prev_r+1].
    #   Then, in the block of indices from prev_r+1 to kth_r (both inclusive),
    #   we place ones for the first (r_K - l_K + 1) positions and zeros for the rest.
    
    ones_count = kth_r - kth_l + 1
    res = list(S)
    start_region = prev_r + 1  # starting index of the region that will be modified
    # Replace positions start_region to start_region+ones_count-1 with '1'
    for i in range(start_region, start_region + ones_count):
        if i <= kth_r:
            res[i] = '1'
    # Replace positions start_region+ones_count to kth_r with '0'
    for i in range(start_region + ones_count, kth_r + 1):
        res[i] = '0'
        
    sys.stdout.write("".join(res))
    
if __name__ == '__main__':
    main()