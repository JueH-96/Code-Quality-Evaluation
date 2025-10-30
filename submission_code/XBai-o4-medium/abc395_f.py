import sys

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    
    U = []
    D = []
    for _ in range(N):
        u = int(input[ptr])
        ptr += 1
        d = int(input[ptr])
        ptr += 1
        U.append(u)
        D.append(d)
    
    per_i_sum = [U[i] + D[i] for i in range(N)]
    H_max_initial = min(per_i_sum)
    total_sum = sum(per_i_sum)
    
    def is_possible(H):
        current_low = max(H - D[0], 0)
        current_high = min(U[0], H)
        if current_low > current_high:
            return False
        for i in range(1, N):
            low_i = max(H - D[i], 0)
            high_i = min(U[i], H)
            new_low = max(low_i, current_low - X)
            new_high = min(high_i, current_high + X)
            if new_low > new_high:
                return False
            current_low, current_high = new_low, new_high
        return True
    
    low = 0
    high = H_max_initial
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(total_sum - ans * N)

if __name__ == '__main__':
    main()