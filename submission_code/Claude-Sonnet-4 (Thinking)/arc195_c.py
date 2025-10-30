def solve():
    T = int(input())
    for _ in range(T):
        R, B = map(int, input().split())
        
        # Impossible cases
        if R == 1 and B == 1:
            print("No")
        elif R == 1 and B >= 2:
            print("No") 
        elif R >= 3 and B == 1:
            print("No")
        else:
            print("Yes")
            pieces = []
            
            if R == 0:  # Blue only
                # Blue pieces form cycle on same parity squares
                for i in range(B):
                    if i == 0:
                        pieces.append(('B', 1, 1))
                    elif i == 1:
                        pieces.append(('B', 2, 2))
                    else:
                        # Add more blue pieces maintaining connectivity
                        if i % 2 == 0:
                            pieces.append(('B', 1, 2*i-1))
                        else:
                            pieces.append(('B', 2, 2*i-2))
                            
            elif B == 0:  # Red only
                if R == 2:
                    pieces = [('R', 1, 1), ('R', 1, 2)]
                elif R == 3:
                    pieces = [('R', 1, 1), ('R', 1, 2), ('R', 2, 2)]
                else:  # R >= 4, use rectangular cycle
                    # Build a rectangular path
                    side = (R + 3) // 4
                    pos = 1
                    # Top side
                    for i in range(min(side, R)):
                        pieces.append(('R', 1, 1 + i))
                        pos += 1
                        if pos > R: break
                    # Right side  
                    for i in range(1, min(side, R - pos + 1)):
                        pieces.append(('R', 1 + i, side))
                        pos += 1
                        if pos > R: break
                    # Bottom side
                    for i in range(min(side-1, R - pos + 1)):
                        pieces.append(('R', side, side - 1 - i))
                        pos += 1
                        if pos > R: break
                    # Left side
                    for i in range(min(side-2, R - pos + 1)):
                        pieces.append(('R', side - 1 - i, 1))
                        pos += 1
                        if pos > R: break
                        
            elif R == 2 and B == 1:
                # Special case that works
                pieces = [('R', 1, 1), ('B', 1, 2), ('R', 2, 1)]
                
            else:  # R >= 2, B >= 2, mixed cycle
                # Use pattern similar to sample: alternate segments
                pieces = []
                # Place blue pieces first
                for i in range(B):
                    if i == 0:
                        pieces.append(('B', 2, 3))
                    elif i == 1:
                        pieces.append(('B', 2, 2))
                    elif i == 2:
                        pieces.append(('B', 3, 3))
                    else:
                        pieces.append(('B', 1 + (i-3) % 2, 4 + (i-3)//2))
                
                # Place red pieces
                for i in range(R):
                    if i == 0:
                        pieces.append(('R', 3, 2))
                    elif i == 1:
                        pieces.append(('R', 2, 4))
                    else:
                        pieces.append(('R', 4 + (i-2) % 2, 1 + (i-2)//2))
            
            for piece_type, r, c in pieces:
                print(f"{piece_type} {r} {c}")

solve()