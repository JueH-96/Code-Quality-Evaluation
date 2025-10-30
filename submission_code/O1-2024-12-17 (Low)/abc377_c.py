def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    occupied = set()
    idx = 2
    for _ in range(M):
        a = int(input_data[idx])
        b = int(input_data[idx+1])
        idx += 2
        occupied.add((a, b))
    
    # Relative knight moves
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    threatened = set()
    
    # For each piece, compute squares that it threatens
    for (r, c) in occupied:
        for dr, dc in knight_moves:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= N and 1 <= nc <= N:
                threatened.add((nr, nc))
    
    # Count of squares that are not safe is the count of unique squares
    # that are either occupied or threatened
    not_safe_squares = len(occupied) + len(threatened)  # but we must subtract overlaps
    # Because threatened and occupied may overlap
    overlap = len(occupied.intersection(threatened))
    not_safe_squares -= overlap
    
    # Total squares = N*N (use Python's arbitrary precision)
    total_squares = N*N
    
    # Safe squares = total_squares - not_safe_squares
    print(total_squares - not_safe_squares)

# Do not forget to call main
if __name__ == "__main__":
    main()