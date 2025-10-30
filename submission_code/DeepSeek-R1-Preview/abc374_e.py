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
    
    def minimal_cost(A, P, B, Q, C):
        if C == 0:
            return 0
        
        cost_S = ((C + A - 1) // A) * P
        cost_T = ((C + B - 1) // B) * Q
        
        s_floor = C // A
        remaining = C - s_floor * A
        if remaining <= 0:
            combo1 = s_floor * P
        else:
            t = (remaining + B - 1) // B
            combo1 = s_floor * P + t * Q
        
        s_plus1 = s_floor + 1
        combo2 = s_plus1 * P
        
        s_minus1 = s_floor - 1
        if s_minus1 >= 0:
            rem = C - s_minus1 * A
            if rem <= 0:
                combo3 = s_minus1 * P
            else:
                t = (rem + B - 1) // B
                combo3 = s_minus1 * P + t * Q
        else:
            combo3 = float('inf')
        
        return min(cost_S, cost_T, combo1, combo2, combo3)
    
    def is_possible(C):
        total = 0
        for A, P, B, Q in processes:
            cost = minimal_cost(A, P, B, Q, C)
            total += cost
            if total > X:
                return False
        return total <= X
    
    low = 0
    high = 10**18
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == '__main__':
    main()