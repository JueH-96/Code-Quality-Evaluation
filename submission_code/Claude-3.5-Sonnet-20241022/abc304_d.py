def solve():
    W, H = map(int, input().split())
    N = int(input())
    
    # Read strawberry positions
    strawberries = []
    for _ in range(N):
        x, y = map(int, input().split())
        strawberries.append((x, y))
    
    # Read cut positions
    A = int(input())
    x_cuts = [0] + list(map(int, input().split())) + [W]
    B = int(input())
    y_cuts = [0] + list(map(int, input().split())) + [H]
    
    # Count strawberries in each piece
    piece_counts = []
    
    # For each rectangular piece
    for i in range(len(x_cuts)-1):
        for j in range(len(y_cuts)-1):
            # Count strawberries in this piece
            count = 0
            x1, x2 = x_cuts[i], x_cuts[i+1]
            y1, y2 = y_cuts[j], y_cuts[j+1]
            
            for sx, sy in strawberries:
                if x1 < sx < x2 and y1 < sy < y2:
                    count += 1
            
            piece_counts.append(count)
    
    # Find min and max counts
    min_count = min(piece_counts)
    max_count = max(piece_counts)
    
    print(min_count, max_count)

solve()