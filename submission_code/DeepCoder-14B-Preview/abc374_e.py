def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    
    processes = []
    index = 2
    for _ in range(N):
        A = int(data[index])
        P = int(data[index+1])
        B = int(data[index+2])
        Q = int(data[index+3])
        processes.append((A, P, B, Q))
        index += 4
    
    def is_possible(C):
        total_cost = 0
        for A, P, B, Q in processes:
            if C == 0:
                # No cost needed
                continue
            s = (C + A - 1) // A  # Equivalent to ceil(C / A)
            cost_s = s * P
            
            t = (C + B - 1) // B  # Equivalent to ceil(C / B)
            cost_t = t * Q
            
            total_cost += min(cost_s, cost_t)
            if total_cost > X:
                return False
        return total_cost <= X
    
    low = 0
    high = 10**18
    best = 0
    
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(best)

if __name__ == '__main__':
    main()