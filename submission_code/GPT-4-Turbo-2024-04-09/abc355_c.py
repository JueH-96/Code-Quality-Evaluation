def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:2+T]))
    
    # Track marked cells
    rows = [0] * N
    cols = [0] * N
    diag1 = 0
    diag2 = 0
    
    # Mapping from number to its grid coordinates
    number_to_coords = {}
    for i in range(1, N+1):
        for j in range(1, N+1):
            number = N * (i - 1) + j
            number_to_coords[number] = (i-1, j-1)
    
    # Process each announced number
    for turn, number in enumerate(A):
        if number in number_to_coords:
            r, c = number_to_coords[number]
            rows[r] += 1
            cols[c] += 1
            
            if r == c:
                diag1 += 1
            if r + c == N - 1:
                diag2 += 1
            
            # Check if bingo is achieved
            if (rows[r] == N or cols[c] == N or diag1 == N or diag2 == N):
                print(turn + 1)
                return
    
    print(-1)

if __name__ == "__main__":
    main()