def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    # We only need a 100x100 grid to mark covered unit squares,
    # since B_i and D_i can go up to 100 (covering squares [0..99] in each dimension).
    covered = [[False] * 100 for _ in range(100)]
    
    idx = 1
    for _ in range(N):
        A, B, C, D = map(int, data[idx:idx+4])
        idx += 4
        
        # Mark each integer "cell" inside the rectangle as covered
        for x in range(A, B):
            for y in range(C, D):
                covered[x][y] = True
    
    # Sum up the covered cells; each cell corresponds to 1 unit of area
    area = sum(sum(row) for row in covered)
    print(area)

# Do not forget to call main
main()