def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    W = int(data[idx])
    H = int(data[idx+1])
    idx += 2
    
    N = int(data[idx])
    idx += 1
    
    strawberries = []
    for _ in range(N):
        p = int(data[idx])
        q = int(data[idx+1])
        strawberries.append((p, q))
        idx += 2
    
    A = int(data[idx])
    idx += 1
    
    a_list = list(map(int, data[idx:idx+A]))
    idx += A
    
    B = int(data[idx])
    idx += 1
    
    b_list = list(map(int, data[idx:idx+B]))
    idx += B
    
    # Determine the x intervals
    a_list_sorted = sorted(a_list)
    x_intervals = []
    prev = 0
    for a in a_list_sorted:
        x_intervals.append((prev, a))
        prev = a
    x_intervals.append((prev, W))
    
    # Determine the y intervals
    b_list_sorted = sorted(b_list)
    y_intervals = []
    prev = 0
    for b in b_list_sorted:
        y_intervals.append((prev, b))
        prev = b
    y_intervals.append((prev, H))
    
    # Create a dictionary to count strawberries in each piece
    from collections import defaultdict
    piece_count = defaultdict(int)
    
    for p, q in strawberries:
        # Find the x interval
        x_idx = 0
        while x_idx < len(x_intervals) and p > x_intervals[x_idx][1]:
            x_idx += 1
        x_interval = x_intervals[x_idx]
        
        # Find the y interval
        y_idx = 0
        while y_idx < len(y_intervals) and q > y_intervals[y_idx][1]:
            y_idx += 1
        y_interval = y_intervals[y_idx]
        
        # Increment the count for this piece
        piece_key = (x_interval, y_interval)
        piece_count[piece_key] += 1
    
    # Find the minimum and maximum counts
    counts = list(piece_count.values())
    if counts:
        m = min(counts)
        M = max(counts)
    else:
        m = 0
        M = 0
    
    # Also, consider pieces with zero strawberries
    total_pieces = (A+1) * (B+1)
    if len(counts) < total_pieces:
        m = 0
    
    print(m, M)

if __name__ == "__main__":
    main()