import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    M = int(input[ptr+1])
    ptr += 2
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    L = list(map(int, input[ptr:ptr+M]))
    ptr += M
    D = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    P.sort()
    sum_P = sum(P)
    
    coupons = list(zip(D, L))
    coupons.sort(key=lambda x: (-x[0], x[1]))
    
    parent = list(range(N + 1))  # 0..N
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    
    total_discount = 0
    for d, l in coupons:
        idx = bisect.bisect_left(P, l)
        pos = find(idx)
        if pos < N:
            total_discount += d
            next_pos = find(pos + 1)
            parent[pos] = next_pos
    
    print(sum_P - total_discount)

if __name__ == "__main__":
    main()