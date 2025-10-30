def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    index = 1
    results = []
    
    for _ in range(T):
        R = int(data[index])
        B = int(data[index + 1])
        index += 2
        
        # We need to find a way to place R reds and B blues in a cycle
        if R == 0 or B == 0:
            # If there are only reds or only blues, they cannot form a cycle
            results.append("No")
            continue
        
        # We can always form a cycle with at least one red and one blue
        # We will use a simple pattern starting at (1,1) and extending right and down
        pieces = []
        r, c = 1, 1
        reds_placed = 0
        blues_placed = 0
        
        # Start placing reds and blues alternately in a cycle
        while reds_placed < R or blues_placed < B:
            if reds_placed < R:
                pieces.append(f"R {r} {c}")
                reds_placed += 1
                r += 1
                c += 1
            if blues_placed < B:
                pieces.append(f"B {r} {c}")
                blues_placed += 1
                r += 1
                c -= 1
        
        # Ensure the last piece can move to the first piece
        if (reds_placed == R and blues_placed == B):
            results.append("Yes")
            results.extend(pieces)
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")