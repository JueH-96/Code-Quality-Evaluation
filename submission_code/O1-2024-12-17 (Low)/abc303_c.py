def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M, H, K = map(int, data[:4])
    S = data[4]
    coords_data = data[5:]

    items = set()
    idx = 0
    for _ in range(M):
        x = int(coords_data[idx]); y = int(coords_data[idx+1])
        idx += 2
        items.add((x, y))

    # Current position
    x, y = 0, 0
    health = H
    
    for move in S:
        # Consume 1 health to move
        health -= 1
        
        # Check if stunned
        if health < 0:
            print("No")
            return
        
        # Make the move
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        # If item exists here and health < K, consume it
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))

    # If we finish all moves without health going below 0, print Yes
    print("Yes")

# Call main to run the solution
main()