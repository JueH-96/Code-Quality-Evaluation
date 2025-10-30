t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    
    # Find all positions of black cells
    black_positions = []
    for i in range(n):
        if s[i] == 'B':
            black_positions.append(i)
    
    if not black_positions:
        print(0)
        continue
    
    # Greedy approach: start from leftmost black cell and cover k cells
    operations = 0
    i = 0
    
    while i < len(black_positions):
        # Start an operation at the current black cell position
        start_pos = black_positions[i]
        operations += 1
        
        # Skip all black cells that are covered by this operation
        # (from start_pos to start_pos + k - 1)
        while i < len(black_positions) and black_positions[i] < start_pos + k:
            i += 1
    
    print(operations)