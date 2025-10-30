def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    occupied = set()
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        occupied.add((a, b))
    
    captured = set()
    
    # Directions for capturing
    directions = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    for (a, b) in occupied:
        for dx, dy in directions:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                captured.add((x, y))
    
    c_size = len(captured)
    overlap = 0
    for (a, b) in occupied:
        if (a, b) in captured:
            overlap += 1
    
    total_empty = N * N - M
    forbidden_empty = c_size - overlap
    answer = total_empty - forbidden_empty
    print(answer)

if __name__ == '__main__':
    main()