def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    h_list = list(map(int, data[1:1+n]))
    
    total_time = 0
    for h in h_list:
        A = total_time + 1
        C = (A - 1) // 3
        lo, hi = 0, h
        while lo < hi:
            mid = (lo + hi) // 2
            last_T = A + mid - 1
            count_multiples = (last_T // 3) - C
            damage = mid + 2 * count_multiples
            if damage >= h:
                hi = mid
            else:
                lo = mid + 1
        total_time += lo
    print(total_time)

if __name__ == "__main__":
    main()