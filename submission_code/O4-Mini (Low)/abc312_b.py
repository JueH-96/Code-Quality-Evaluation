def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(N)]
    
    # Precompute masks for a 9x9 region
    black_req = [[False]*9 for _ in range(9)]
    white_req = [[False]*9 for _ in range(9)]
    
    # Mark black requirements for TL 3x3 and BR 3x3
    for r in range(3):
        for c in range(3):
            black_req[r][c] = True
    for r in range(6, 9):
        for c in range(6, 9):
            black_req[r][c] = True
    
    # Function to mark adjacent cells around a block
    def mark_adj(block_r0, block_r1, block_c0, block_c1):
        for br in range(block_r0, block_r1+1):
            for bc in range(block_c0, block_c1+1):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = br+dr, bc+dc
                        if 0 <= nr < 9 and 0 <= nc < 9:
                            # skip if inside the block itself
                            if not (block_r0 <= nr <= block_r1 and block_c0 <= nc <= block_c1):
                                white_req[nr][nc] = True
    
    # Mark adjacency for TL block (0..2,0..2) and BR block (6..8,6..8)
    mark_adj(0, 2, 0, 2)
    mark_adj(6, 8, 6, 8)
    
    out = []
    # Slide a 9x9 window
    for i in range(N - 9 + 1):
        for j in range(M - 9 + 1):
            ok = True
            # check all 9x9 cells
            for dr in range(9):
                if not ok:
                    break
                row = grid[i+dr]
                for dc in range(9):
                    ch = row[j+dc]
                    if black_req[dr][dc]:
                        if ch != '#':
                            ok = False
                            break
                    if white_req[dr][dc]:
                        if ch != '.':
                            ok = False
                            break
            if ok:
                # report 1-based coordinates
                out.append(f"{i+1} {j+1}")
    
    # print results in lex order (they naturally are)
    print("
".join(out))


if __name__ == "__main__":
    main()