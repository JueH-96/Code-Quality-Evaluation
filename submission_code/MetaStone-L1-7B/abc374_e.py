import math

def compute_min_cost(A, P, B, Q, W):
    if W == 0:
        return 0
    
    # Calculate cost for all S_i
    s = (W + A - 1) // A
    cost_s = s * P
    
    # Calculate cost for all T_i
    t = (W + B - 1) // B
    cost_t = t * Q
    
    if cost_s == 0 and cost_t == 0:
        return 0
    
    denominator = A * Q + B * P
    if denominator == 0:
        return max(cost_s, cost_t)
    
    s_real = (W * Q) / denominator
    
    s_candidates = {math.floor(s_real), math.ceil(s_real), math.floor(s_real) - 1, math.ceil(s_real) + 1}
    s_candidates.add(0)
    s_candidates.add(1)
    
    min_cost = min(cost_s, cost_t)
    
    for s in s_candidates:
        if s < 0:
            continue
        required = W - s * A
        if required <= 0:
            t = 0
        else:
            t = (required + B - 1) // B
        cost = s * P + t * Q
        if cost < min_cost:
            min_cost = cost
    
    return min_cost

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    
    processes = []
    for _ in range(N):
        A = int(input[idx])
        idx += 1
        P = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        Q = int(input[idx])
        idx += 1
        processes.append((A, P, B, Q))
    
    low = 0
    high = 10**18
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        total = 0
        possible = True
        
        for (A, P, B, Q) in processes:
            if mid == 0:
                continue
            min_cost = compute_min_cost(A, P, B, Q, mid)
            if min_cost > X:
                possible = False
                break
            total += min_cost
        
        if possible:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == '__main__':
    main()