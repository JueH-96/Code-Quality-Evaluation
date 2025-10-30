def can_capture(x1, y1, x2, y2):
    # Check if piece at (x1,y1) can capture piece at (x2,y2)
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

def get_threatened_squares(N, pieces):
    # Set to store squares that are threatened (can be captured)
    threatened = set()
    
    # For each piece
    for x, y in pieces:
        # Add the square itself
        threatened.add((x, y))
        
        # Check all squares that could capture this piece
        moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dx, dy in moves:
            new_x = x + dx
            new_y = y + dy
            # If the square is within bounds, it's threatened
            if 1 <= new_x <= N and 1 <= new_y <= N:
                threatened.add((new_x, new_y))
    
    return threatened

def solve():
    # Read input
    N, M = map(int, input().split())
    pieces = []
    for _ in range(M):
        a, b = map(int, input().split())
        pieces.append((a, b))
    
    # Get all threatened squares
    threatened = get_threatened_squares(N, pieces)
    
    # Total squares minus threatened squares is the answer
    total_squares = N * N
    threatened_count = len(threatened)
    
    print(total_squares - threatened_count)

# Run the solution
solve()