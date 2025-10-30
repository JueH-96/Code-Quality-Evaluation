import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    m = int(input[idx])
    idx += 1
    
    p = list(map(int, input[idx:idx+n]))
    idx += n
    p.sort()
    
    l_list = list(map(int, input[idx:idx+m]))
    idx += m
    d_list = list(map(int, input[idx:idx+m]))
    idx += m
    
    # Create list of coupons (L, D) and sort by D descending
    coupons = list(zip(l_list, d_list))
    coupons.sort(key=lambda x: -x[1])
    
    used = [False] * n
    total_discount = 0
    
    for l, d in coupons:
        # Find the first index where p[i] >= l using bisect_left
        idx_p = bisect.bisect_left(p, l)
        # Iterate from idx_p to find the first unused item
        for i in range(idx_p, n):
            if not used[i]:
                used[i] = True
                total_discount += d
                break
    
    minimal_total = sum(p) - total_discount
    print(minimal_total)

if __name__ == '__main__':
    main()