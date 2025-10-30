import bisect

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
    a = list(map(int, data[idx:idx+A]))
    idx += A
    
    B = int(data[idx])
    idx += 1
    b = list(map(int, data[idx:idx+B]))
    idx += B
    
    from collections import defaultdict
    counter = defaultdict(int)
    
    for p, q in strawberries:
        v = bisect.bisect_left(a, p)
        h = bisect.bisect_left(b, q)
        counter[(v, h)] += 1
    
    counts = list(counter.values())
    M = max(counts) if counts else 0
    
    total_cells = (A + 1) * (B + 1)
    num_non_empty = len(counts)
    
    if num_non_empty < total_cells:
        m = 0
    else:
        m = min(counts) if counts else 0
    
    print(m, M)

if __name__ == "__main__":
    main()