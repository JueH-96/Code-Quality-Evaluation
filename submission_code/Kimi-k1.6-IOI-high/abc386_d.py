import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    
    B = []
    W = []
    for _ in range(m):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        c = data[idx]
        idx += 1
        if c == 'B':
            B.append((x, y))
        else:
            W.append((x, y))
    
    # Process B cells to compute max_y_suffix
    sorted_B = sorted(B, key=lambda p: p[0])
    sorted_x_B = [p[0] for p in sorted_B]
    max_y_suffix = []
    if sorted_B:
        max_y_suffix = [0] * len(sorted_B)
        max_y_suffix[-1] = sorted_B[-1][1]
        for i in range(len(sorted_B)-2, -1, -1):
            max_y_suffix[i] = max(sorted_B[i][1], max_y_suffix[i+1])
    
    valid = True
    
    # Check all W cells
    for a, b in W:
        idx_bisect = bisect.bisect_left(sorted_x_B, a)
        if idx_bisect < len(sorted_x_B):
            if max_y_suffix[idx_bisect] >= b:
                valid = False
                break
    
    if valid:
        # Process W cells to compute min_b_prefix
        sorted_W = sorted(W, key=lambda p: p[0])
        sorted_a_W = [p[0] for p in sorted_W]
        min_b_prefix = []
        if sorted_W:
            min_b_prefix = [sorted_W[0][1]]
            for i in range(1, len(sorted_W)):
                min_b_prefix.append(min(sorted_W[i][1], min_b_prefix[-1]))
        
        # Check all B cells
        for x, y in B:
            idx_bisect = bisect.bisect_right(sorted_a_W, x) - 1
            if idx_bisect >= 0:
                if min_b_prefix[idx_bisect] <= y:
                    valid = False
                    break
    
    print("Yes" if valid else "No")

if __name__ == "__main__":
    main()