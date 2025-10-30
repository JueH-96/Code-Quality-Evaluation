def solve():
    n, m = map(int, input().split())
    occupied_positions = set()
    for _ in range(m):
        u, v = map(int, input().split())
        occupied_positions.add((u, v))
    
    captured_squares = set()
    capture_offsets = [
        (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    for piece_pos in occupied_positions:
        r, c = piece_pos
        for dr, dc in capture_offsets:
            captured_r, captured_c = r + dr, c + dc
            if 1 <= captured_r <= n and 1 <= captured_c <= n:
                captured_squares.add((captured_r, captured_c))
                
    num_captured = len(captured_squares)
    num_safe_empty_squares = n**2 - m - num_captured
    print(num_safe_empty_squares)

if __name__ == '__main__':
    solve()