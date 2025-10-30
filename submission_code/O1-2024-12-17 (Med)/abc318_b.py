def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = input_data[1:]
    
    # We can safely create a 101x101 grid since 0 <= A_i, B_i, C_i, D_i <= 100.
    # coverage[x][y] will be 1 if the unit square with corner (x,y) is covered by any sheet.
    coverage = [[0]*101 for _ in range(101)]
    
    idx = 0
    for _ in range(N):
        A = int(coords[idx]); B = int(coords[idx+1])
        C = int(coords[idx+2]); D = int(coords[idx+3])
        idx += 4
        
        for x in range(A, B):
            for y in range(C, D):
                coverage[x][y] = 1
    
    # The area is the sum of all covered unit squares.
    area = sum(sum(row) for row in coverage)
    print(area)

# Do not forget to call main()
if __name__ == "__main__":
    main()