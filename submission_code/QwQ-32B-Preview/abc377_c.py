def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    # Read all a_k and b_k
    pieces = []
    for i in range(M):
        a = int(data[2 + 2*i])
        b = int(data[3 + 2*i])
        pieces.append((a, b))
    
    # Define capture offsets
    offsets = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Set to store unique dangerous squares
    dangerous = set()
    
    for a, b in pieces:
        for dx, dy in offsets:
            i = a + dx
            j = b + dy
            if 1 <= i <= N and 1 <= j <= N:
                dangerous.add((i, j))
    
    # Total squares
    total_squares = N * N
    
    # Empty squares
    empty_squares = total_squares - M
    
    # Unique dangerous squares
    dangerous_count = len(dangerous)
    
    # Safe squares
    safe_squares = empty_squares - dangerous_count
    
    print(safe_squares)

if __name__ == "__main__":
    main()