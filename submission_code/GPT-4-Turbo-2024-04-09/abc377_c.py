def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    occupied = set()
    threatened = set()
    
    # Read all pieces positions
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        occupied.add((a, b))
        
        # Calculate all threatened positions by this piece
        moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        for dx, dy in moves:
            nx, ny = a + dx, b + dy
            if 1 <= nx <= N and 1 <= ny <= N:
                threatened.add((nx, ny))
    
    # Total squares in the grid
    total_squares = N * N
    
    # Squares that are either occupied or threatened
    non_safe_squares = len(occupied.union(threatened))
    
    # Safe squares where you can place your piece
    safe_squares = total_squares - non_safe_squares
    
    print(safe_squares)