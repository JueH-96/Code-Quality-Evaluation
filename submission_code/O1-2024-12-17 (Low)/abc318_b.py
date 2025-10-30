def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    sheets = input_data[1:]
    
    # Initialize a grid with False (no coverage) up to 101 x 101
    coverage = [[False]*101 for _ in range(101)]
    
    idx = 0
    for _ in range(N):
        A = int(sheets[idx]); B = int(sheets[idx+1])
        C = int(sheets[idx+2]); D = int(sheets[idx+3])
        idx += 4
        
        # Mark coverage for each 1x1 unit
        for x in range(A, B):
            for y in range(C, D):
                coverage[x][y] = True
    
    # Calculate covered area
    area = 0
    for x in range(101):
        for y in range(101):
            if coverage[x][y]:
                area += 1

    print(area)

# Do not forget to call main()
main()