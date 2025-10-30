def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    # Convert the string for each stand into a bitmask representation.
    # Each bit in the bitmask represents a flavor (1 if the flavor is sold, 0 otherwise).
    stand_masks = []
    for _ in range(N):
        s = data[idx]
        idx += 1
        mask = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                mask |= (1 << j)
        stand_masks.append(mask)
    
    # The target bitmask when all flavors are obtained.
    target = (1 << M) - 1
    
    # Considering the small constraints (N, M <= 10), we can use bitmask subset enumeration.
    ans = N + 1  # Initialize ans to a large number (greater than maximum possible stands)
    for subset in range(1, 1 << N):
        current_mask = 0
        count = 0
        # For each stand included in the subset, update the current_mask with its flavors.
        for stand in range(N):
            if subset & (1 << stand):
                current_mask |= stand_masks[stand]
                count += 1
        # If the union of flavors covers all flavors, update the answer.
        if current_mask == target:
            ans = min(ans, count)
    
    print(ans)

if __name__ == '__main__':
    main()