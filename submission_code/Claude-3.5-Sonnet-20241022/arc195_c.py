def can_move(p1, p2, color):
    r1, c1 = p1
    r2, c2 = p2
    if color == 'R':  # Red piece - moves horizontally or vertically
        return (abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1)
    else:  # Blue piece - moves diagonally
        return abs(r1 - r2) == 1 and abs(c1 - c2) == 1

def solve_case(R, B):
    total = R + B
    if total < 2:
        return None
        
    # For R >= 4, we can make a square with red pieces
    if R >= 4:
        pieces = []
        pieces.append(('R', 1, 1))
        pieces.append(('R', 1, 2))
        pieces.append(('R', 2, 2))
        pieces.append(('R', 2, 1))
        
        # Add remaining pieces in a zigzag pattern
        curr_r, curr_c = 2, 1
        remaining = R + B - 4
        for i in range(remaining):
            if len(pieces) % 2 == 0:
                curr_c += 1
            else:
                curr_r += 1
            if i < R - 4:
                pieces.append(('R', curr_r, curr_c))
            else:
                pieces.append(('B', curr_r, curr_c))
        return pieces
        
    # For B >= 4, we can make a square with blue pieces
    elif B >= 4:
        pieces = []
        pieces.append(('B', 1, 1))
        pieces.append(('B', 2, 2))
        pieces.append(('B', 3, 1))
        pieces.append(('B', 2, 0))
        
        # Add remaining pieces
        curr_r, curr_c = 2, 0
        remaining = R + B - 4
        for i in range(remaining):
            if len(pieces) % 2 == 0:
                curr_c += 2
            else:
                curr_r += 2
            if i < B - 4:
                pieces.append(('B', curr_r, curr_c))
            else:
                pieces.append(('R', curr_r, curr_c))
        return pieces
        
    # Special cases for R + B >= 2
    elif R == 2 and B >= 1:
        return [('B', 2, 3), ('R', 3, 2), ('B', 2, 2)]
    elif B == 2 and R >= 1:
        return [('R', 2, 2), ('B', 1, 1), ('B', 2, 0)]
    elif R == 1 and B == 1:
        return None
    elif B == 3 and R == 2:
        return [('B', 2, 3), ('R', 3, 2), ('B', 2, 2), ('B', 3, 3), ('R', 2, 4)]
        
    return None

def main():
    T = int(input())
    for _ in range(T):
        R, B = map(int, input().split())
        result = solve_case(R, B)
        
        if result is None:
            print("No")
        else:
            print("Yes")
            for color, r, c in result:
                print(f"{color} {r} {c}")

if __name__ == "__main__":
    main()