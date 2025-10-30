# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    rows = set()
    cols = set()
    main_diagonals = set()
    anti_diagonals = set()
    
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        rows.add(a)
        cols.add(b)
        main_diagonals.add(a + b)
        anti_diagonals.add(a - b)
    
    total_squares = N * N
    captured_squares = (len(rows) * N) + (len(cols) * N) + (len(main_diagonals) * N) + (len(anti_diagonals) * N)
    
    # Subtract the intersections
    captured_squares -= len(rows) * len(cols)
    captured_squares -= len(rows) * len(main_diagonals)
    captured_squares -= len(rows) * len(anti_diagonals)
    captured_squares -= len(cols) * len(main_diagonals)
    captured_squares -= len(cols) * len(anti_diagonals)
    captured_squares -= len(main_diagonals) * len(anti_diagonals)
    
    # Add back the triple intersections
    captured_squares += len(rows) * len(cols) * len(main_diagonals)
    captured_squares += len(rows) * len(cols) * len(anti_diagonals)
    captured_squares += len(rows) * len(main_diagonals) * len(anti_diagonals)
    captured_squares += len(cols) * len(main_diagonals) * len(anti_diagonals)
    
    # Add back the quadruple intersections
    captured_squares -= len(rows) * len(cols) * len(main_diagonals) * len(anti_diagonals)
    
    safe_squares = total_squares - captured_squares
    print(safe_squares)

if __name__ == "__main__":
    main()