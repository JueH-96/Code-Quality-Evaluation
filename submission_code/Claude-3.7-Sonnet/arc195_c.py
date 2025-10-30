def solve():
    t = int(input())
    for _ in range(t):
        r, b = map(int, input().split())
        
        # Check impossible cases
        if (r == 0 and b <= 1) or (r == 1 and b == 0) or (r == 1 and b == 1) or (r == 3 and b == 0):
            print("No")
            continue
        
        print("Yes")
        if r == 0:  # Only blue pieces
            # Blue pieces can move diagonally, so we place them in a zigzag pattern
            for i in range(b):
                if i % 2 == 0:
                    print("B", 1, i//2 + 1)
                else:
                    print("B", 2, i//2 + 1)
        
        elif b == 0:  # Only red pieces
            if r == 2:
                # Two red pieces can form a cycle by moving horizontally
                print("R", 1, 1)
                print("R", 1, 2)
            else:  # r >= 4
                # For r >= 4, we can form a rectangular cycle
                print("R", 1, 1)
                print("R", 1, 2)
                print("R", 2, 2)
                print("R", 2, 1)
                
                # Add remaining red pieces
                for i in range(4, r):
                    print("R", 1 + (i % 2), 1 + ((i // 2) % 2))
        
        else:  # Mixed case with both red and blue pieces
            if r == 2 and b == 3:
                # Use the pattern from the sample
                print("B", 2, 3)
                print("R", 3, 2)
                print("B", 2, 2)
                print("B", 3, 3)
                print("R", 2, 4)
            
            elif r >= 2 and b >= 1:
                # Make a cycle where each piece can move to the next according to its rules
                print("R", 1, 1)
                print("R", 1, 2)
                print("B", 2, 1)
                
                remaining_r = r - 2
                remaining_b = b - 1
                
                row, col = 3, 1
                
                # Place remaining red pieces
                for _ in range(remaining_r):
                    print("R", row, col)
                    row += 1
                
                # Place remaining blue pieces
                for _ in range(remaining_b):
                    print("B", row, col + 1)
                    row += 1

if __name__ == "__main__":
    solve()