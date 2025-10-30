def calculate_t():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    T = 0
    for h in H:
        B = T % 3
        low = 1
        high = h
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            c = (B + mid) // 3
            sum_ = 2 * c + mid
            if sum_ >= h:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        T += ans
    print(T)

calculate_t()