def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, H, K = map(int, input_data[:4])
    S = input_data[4]
    
    # Read item coordinates
    items = set()
    idx = 5
    for _ in range(M):
        x_i = int(input_data[idx]); y_i = int(input_data[idx+1])
        idx += 2
        items.add((x_i, y_i))
    
    x, y = 0, 0  # Starting coordinates
    
    for move in S:
        # Make the move
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        # Decrease health
        H -= 1
        
        # Check if collapsed
        if H < 0:
            print("No")
            return
        
        # If there is an item and health < K, consume it
        if (x, y) in items and H < K:
            H = K
            items.remove((x, y))
    
    # If completed all moves without collapsing
    print("Yes")

# Do not forget to call main()
main()