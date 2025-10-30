import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    X = int(data[idx + 1])
    idx += 2
    
    U = []
    D = []
    sum_UD = 0
    H_max = float('inf')
    
    for _ in range(N):
        u = int(data[idx])
        d = int(data[idx + 1])
        idx += 2
        U.append(u)
        D.append(d)
        s = u + d
        sum_UD += s
        if s < H_max:
            H_max = s
    
    def feasible(H):
        curr_low = -1
        curr_high = -1
        for i in range(N):
            u = U[i]
            d = D[i]
            lower_i = max(0, H - d)
            upper_i = min(u, H)
            if lower_i > upper_i:
                return False
            if i == 0:
                curr_low = lower_i
                curr_high = upper_i
            else:
                new_low = max(lower_i, curr_low - X)
                new_high = min(upper_i, curr_high + X)
                if new_low > new_high:
                    return False
                curr_low = new_low
                curr_high = new_high
        return True
    
    low = 0
    high = H_max
    best = 0
    
    while low <= high:
        mid = (low + high) // 2
        if feasible(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    cost = sum_UD - best * N
    print(cost)

if __name__ == "__main__":
    main()