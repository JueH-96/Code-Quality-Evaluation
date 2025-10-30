import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    P = list(map(int, data[2:2+n]))
    P.sort()
    
    coupons = []
    index = 2 + n
    for _ in range(m):
        L = int(data[index])
        D = int(data[index + 1])
        coupons.append((D, L))
        index += 2
    
    # Sort coupons in descending order of D
    coupons.sort(reverse=True, key=lambda x: x[0])
    
    discount = 0
    P_list = P.copy()
    
    for D, L in coupons:
        idx = bisect.bisect_left(P_list, L)
        if idx < len(P_list):
            discount += D
            del P_list[idx]
    
    total = sum(P) - discount
    print(total)

if __name__ == '__main__':
    main()