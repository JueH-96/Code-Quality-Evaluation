def solve():
    n, m = map(int, input().split())
    occupied_squares = set()
    piece_positions = []
    for _ in range(m):
        r, c = map(int, input().split())
        occupied_squares.add((r, c))
        piece_positions.append((r, c))
    
    attacked_squares = set()
    capture_offsets = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    for r0, c0 in piece_positions:
        for dr, dc in capture_offsets:
            r, c = r0 + dr, c0 + dc
            if 1 <= r <= n and 1 <= c <= n:
                attacked_squares.add((r, c))
                
    size_attacked = len(attacked_squares)
    intersection_squares = occupied_squares.intersection(attacked_squares)
    size_intersection = len(intersection_squares)
    
    unsafe_squares_count = m + size_attacked - size_intersection
    safe_squares_count = n**2 - unsafe_squares_count
    print(safe_squares_count)

if __name__ == '__main__':
    solve()