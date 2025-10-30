def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    pieces = []
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        pieces.append((a, b))
        index += 2
    
    # Knight's move offsets
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Set to store all capturable positions
    capturable_positions = set()
    
    for a, b in pieces:
        for dx, dy in knight_moves:
            x, y = a + dx, b + dy
            if 1 <= x <= N and 1 <= y <= N:
                capturable_positions.add((x, y))
    
    # Total number of positions
    total_positions = N * N
    
    # Number of positions that are occupied by pieces
    occupied_positions = M
    
    # Number of positions that are capturable
    capturable_positions_count = len(capturable_positions)
    
    # Safe positions are those not occupied and not capturable
    safe_positions = total_positions - occupied_positions - capturable_positions_count
    
    print(safe_positions)