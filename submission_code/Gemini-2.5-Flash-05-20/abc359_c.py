import sys

def solve():
    S_x, S_y = map(int, sys.stdin.readline().split())
    T_x, T_y = map(int, sys.stdin.readline().split())

    dx = abs(S_x - T_x)
    dy = abs(S_y - T_y)

    ans = 0

    if dx % 2 == 0 and dy % 2 == 0:
        # Both dx and dy are even.
        # This implies S_x and T_x have same parity, S_y and T_y have same parity.
        # e.g., (E,E) to (E,E) or (O,O) to (O,O)
        ans = (dx + dy) // 2
    elif dx % 2 == 1 and dy % 2 == 1:
        # Both dx and dy are odd.
        # This implies S_x and T_x have different parities, S_y and T_y have different parities.
        # e.g., (E,E) to (O,O) or (O,E) to (E,O)
        ans = max(dx, dy)
    elif dx % 2 == 1 and dy % 2 == 0:
        # dx is odd, dy is even.
        # This implies S_x and T_x have different parities, S_y and T_y have same parities.
        # e.g., (E,E) to (O,E)
        ans = dy
    else: # dx % 2 == 0 and dy % 2 == 1
        # dx is even, dy is odd.
        # This implies S_x and T_x have same parities, S_y and T_y have different parities.
        # e.g., (E,E) to (E,O)
        ans = dx
    
    print(ans)

solve()