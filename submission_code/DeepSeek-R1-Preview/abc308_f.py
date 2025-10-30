import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    m = int(data[idx+1])
    idx +=2
    p = list(map(int, data[idx:idx+n]))
    idx +=n
    l = list(map(int, data[idx:idx+m]))
    idx +=m
    d = list(map(int, data[idx:idx+m]))
    
    p.sort()
    coupons = list(zip(l, d))
    coupons.sort(key=lambda x: -x[1])  # Sort by D descending
    
    parent = list(range(n))
    
    def find(i):
        while True:
            if i >= n:
                return n
            if parent[i] == i:
                return i
            # Path compression
            parent[i] = parent[parent[i]]
            i = parent[i]
    
    total_discount = 0
    
    for li, di in coupons:
        start = bisect.bisect_left(p, li)
        if start >= n:
            continue
        pos = find(start)
        if pos < n:
            total_discount += di
            parent[pos] = find(pos + 1)
    
    print(sum(p) - total_discount)

if __name__ == "__main__":
    main()