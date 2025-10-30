def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    stands = input_data[2:]
    
    # Precompute the bit mask representation for each stand's available flavors
    stand_masks = []
    for s in stands:
        mask = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                mask |= (1 << j)
        stand_masks.append(mask)
    
    # Full mask where all M flavors are represented (bit j set means flavor j is obtained)
    full_mask = (1 << M) - 1
    min_stands = N + 1  # Initialize with a value larger than maximum possible stands
    
    # There are at most 2^N subsets. Try every subset.
    for subset in range(1, 1 << N):
        current_mask = 0
        count = 0
        for stand_index in range(N):
            if (subset >> stand_index) & 1:
                current_mask |= stand_masks[stand_index]
                count += 1
        # If this subset covers all flavors, update the min_stands if needed.
        if current_mask == full_mask:
            min_stands = min(min_stands, count)
    
    print(min_stands)

if __name__ == '__main__':
    main()