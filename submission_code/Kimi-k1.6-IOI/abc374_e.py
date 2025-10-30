def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    steps = []
    for _ in range(N):
        A = int(input[idx])
        P = int(input[idx+1])
        B = int(input[idx+2])
        Q = int(input[idx+3])
        steps.append((A, P, B, Q))
        idx +=4
    
    low = 0
    high = 10**18
    best = 0
    
    while low <= high:
        mid = (low + high) // 2
        total_cost = 0
        feasible = True
        for (A, P, B, Q) in steps:
            if mid == 0:
                cost = 0
            else:
                # Compute cost for only S
                a_s = (mid + A - 1) // A
                cost_s = a_s * P
                # Compute cost for only T
                b_t = (mid + B - 1) // B
                cost_t = b_t * Q
                min_c = min(cost_s, cost_t)
                
                # Check a from 0 to 3
                for a in range(0, 4):
                    total_a = a * A
                    if total_a >= mid:
                        c = a * P
                    else:
                        rem = mid - total_a
                        b = (rem + B - 1) // B
                        c = a * P + b * Q
                    if c < min_c:
                        min_c = c
                
                # Check a around a_ceil
                a_ceil = a_s
                a_min = max(0, a_ceil - 3)
                a_max = a_ceil + 3
                for a in range(a_min, a_max + 1):
                    total_a = a * A
                    if total_a >= mid:
                        c = a * P
                    else:
                        rem = mid - total_a
                        if rem <= 0:
                            b = 0
                        else:
                            b = (rem + B - 1) // B
                        c = a * P + b * Q
                    if c < min_c:
                        min_c = c
                cost = min_c
            total_cost += cost
            if total_cost > X:
                feasible = False
                break
        if feasible:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    print(best)

if __name__ == "__main__":
    main()