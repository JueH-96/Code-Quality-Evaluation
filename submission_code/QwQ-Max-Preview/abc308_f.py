import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    L = list(map(int, input[ptr:ptr+M]))
    ptr += M
    D = list(map(int, input[ptr:ptr+M]))
    
    P.sort()
    coupons = sorted(zip(L, D), key=lambda x: (-x[1], x[0]))
    
    parent = list(range(N + 1))  # parent[i] is the next available index >= i
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    total_discount = 0
    for L_i, D_i in coupons:
        j = bisect.bisect_left(P, L_i)
        k = find(j)
        if k < N:
            total_discount += D_i
            parent[k] = find(k + 1)
    
    sum_p = sum(P)
    print(sum_p - total_discount)

if __name__ == "__main__":
    main()