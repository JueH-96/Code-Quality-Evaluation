def solve():
    n, m = map(int, input().split())
    pieces_positions = []
    for _ in range(m):
        pieces_positions.append(tuple(map(int, input().split())))
    
    occupied_squares = set(pieces_positions)
    safe_squares_count = 0
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i, j) in occupied_squares:
                continue
            is_safe = True
            for piece_pos in pieces_positions:
                a_k, b_k = piece_pos
                if i == a_k or j == b_k or i + j == a_k + b_k or i - j == a_k - b_k:
                    is_safe = False
                    break
            if is_safe:
                safe_squares_count += 1
                
    print(safe_squares_count)

if __name__ == '__main__':
    solve()