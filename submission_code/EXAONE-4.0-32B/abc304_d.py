import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    W = int(data[0])
    H = int(data[1])
    N = int(data[2])
    berries = []
    index = 3
    for i in range(N):
        p = int(data[index])
        q = int(data[index+1])
        index += 2
        berries.append((p, q))
        
    A = int(data[index])
    index += 1
    vert_cuts = list(map(int, data[index:index+A]))
    index += A
    
    B = int(data[index])
    index += 1
    horiz_cuts = list(map(int, data[index:index+B]))
    
    counts = {}
    for (p, q) in berries:
        i_x = bisect.bisect_left(vert_cuts, p)
        i_y = bisect.bisect_left(horiz_cuts, q)
        key = (i_x, i_y)
        counts[key] = counts.get(key, 0) + 1
        
    total_pieces = (A + 1) * (B + 1)
    if len(counts) < total_pieces:
        min_val = 0
    else:
        min_val = min(counts.values())
        
    max_val = max(counts.values()) if counts else 0
    print(f"{min_val} {max_val}")

if __name__ == "__main__":
    main()