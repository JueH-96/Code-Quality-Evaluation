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
        P = int(input[idx+1])
        B = int(input[idx+2])
        Q = int(input[idx+3])
        processes.append((A, P, B, Q))
        idx +=4
    
    low = 0
    high = 10**18
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        total = 0
        feasible = True
        for (A, P, B, Q) in processes:
            # Calculate cost_s
            s = (mid + A -1) // A
            cost_s = s * P
            # Calculate cost_t
            t = (mid + B -1) // B
            cost_t = t * Q
            # Calculate cost_stair
            s_max = mid // A
            remaining = mid - s_max * A
            if remaining <=0:
                t_needed = 0
            else:
                t_needed = (remaining + B -1) // B
            cost_stair = s_max * P + t_needed * Q
            # Calculate cost_stair2
            t_max = mid // B
            remaining = mid - t_max * B
            if remaining <=0:
                s_needed = 0
            else:
                s_needed = (remaining + A -1) // A
            cost_stair2 = s_needed * P + t_max * Q
            
            min_cost = min(cost_s, cost_t, cost_stair, cost_stair2)
            total += min_cost
            if total > X:
                feasible = False
                break
        if feasible:
            ans = mid
            low = mid +1
        else:
            high = mid -1
    print(ans)

if __name__ == '__main__':
    main()