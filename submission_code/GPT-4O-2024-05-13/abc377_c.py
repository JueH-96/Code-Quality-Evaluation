# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    pieces = []
    for i in range(M):
        a = int(data[2 + 2 * i])
        b = int(data[3 + 2 * i])
        pieces.append((a, b))
    
    # Directions a piece can capture another piece
    directions = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Set to keep track of all squares that are threatened
    threatened_squares = set()
    
    for (a, b) in pieces:
        threatened_squares.add((a, b))
        for (dx, dy) in directions:
            x, y = a + dx, b + dy
            if 1 <= x <= N and 1 <= y <= N:
                threatened_squares.add((x, y))
    
    # Total number of squares
    total_squares = N * N
    
    # Number of safe squares
    safe_squares = total_squares - len(threatened_squares)
    
    print(safe_squares)

if __name__ == "__main__":
    main()