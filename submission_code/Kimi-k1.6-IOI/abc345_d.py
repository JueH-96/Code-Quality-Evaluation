def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    tiles = []
    for _ in range(N):
        a = int(input[idx])
        b = int(input[idx+1])
        tiles.append((a, b))
        idx += 2
    
    found = False
    for mask in range(1, 1 << N):
        subset = []
        for i in range(N):
            if (mask & (1 << i)):
                subset.append(tiles[i])
        total = sum(a * b for a, b in subset)
        if total != H * W:
            continue
        
        # Sort subset by descending area
        subset.sort(key=lambda x: -(x[0] * x[1]))
        
        def can_cover():
            def backtrack(covered, used):
                # Find first empty cell
                for pos in range(H * W):
                    if not (covered & (1 << pos)):
                        break
                else:
                    return True
                r = pos // W
                c = pos % W
                
                for i in range(len(subset)):
                    if not (used & (1 << i)):
                        a, b = subset[i]
                        for h, w in [(a, b), (b, a)]:
                            if r + h > H or c + w > W:
                                continue
                            # Check if all cells in the rectangle are uncovered
                            valid = True
                            for dr in range(h):
                                for dc in range(w):
                                    nr = r + dr
                                    nc = c + dc
                                    p = nr * W + nc
                                    if (covered & (1 << p)) != 0:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                # Compute the mask for this rectangle
                                mask_rect = 0
                                for dr in range(h):
                                    for dc in range(w):
                                        mask_rect |= (1 << ((r + dr) * W + (c + dc)))
                                if backtrack(covered | mask_rect, used | (1 << i)):
                                    return True
                return False
            return backtrack(0, 0)
        
        if can_cover():
            found = True
            break
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()