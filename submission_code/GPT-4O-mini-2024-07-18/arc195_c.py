def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        R, B = map(int, data[i].split())
        
        # Check the condition for placement
        if abs(R - B) > 1:
            results.append("No")
            continue
        
        # If we can place them, we will generate a valid configuration
        results.append("Yes")
        placements = []
        
        # Start placing pieces in a zigzag manner
        if R >= B:
            # Start with Red
            for j in range(R):
                placements.append(f"R {j + 1} 1")
            for j in range(B):
                placements.append(f"B {j + 1} 2")
        else:
            # Start with Blue
            for j in range(B):
                placements.append(f"B {j + 1} 1")
            for j in range(R):
                placements.append(f"R {j + 1} 2")
        
        results.extend(placements)
    
    sys.stdout.write("
".join(results) + "
")