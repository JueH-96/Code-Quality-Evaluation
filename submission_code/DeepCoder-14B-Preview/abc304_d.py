from bisect import bisect_left
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    W = int(data[ptr])
    ptr += 1
    H = int(data[ptr])
    ptr += 1
    
    N = int(data[ptr])
    ptr += 1
    
    strawberries = []
    for _ in range(N):
        p = int(data[ptr])
        ptr += 1
        q = int(data[ptr])
        ptr += 1
        strawberries.append((p, q))
    
    A = int(data[ptr])
    ptr += 1
    a = list(map(int, data[ptr:ptr+A]))
    ptr += A
    
    B = int(data[ptr])
    ptr += 1
    b = list(map(int, data[ptr:ptr+B]))
    ptr += B
    
    counts = defaultdict(int)
    
    for p, q in strawberries:
        i = bisect_left(a, p)
        j = bisect_left(b, q)
        counts[(i, j)] += 1
    
    if counts:
        max_count = max(counts.values())
    else:
        max_count = 0
    
    total_pieces = (A + 1) * (B + 1)
    
    if len(counts) < total_pieces:
        min_count = 0
    else:
        min_count = min(counts.values()) if counts else 0
    
    print(min_count, max_count)

if __name__ == "__main__":
    main()