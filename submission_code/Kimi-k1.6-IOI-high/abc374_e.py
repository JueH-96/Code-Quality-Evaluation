def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    X = int(data[idx])
    idx += 1
    processes = []
    for _ in range(N):
        A = int(data[idx])
        P = int(data[idx+1])
        B = int(data[idx+2])
        Q = int(data[idx+3])
        processes.append((A, P, B, Q))
        idx +=4
    
    low = 0
    high = 1 << 60
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        total_cost = 0
        feasible = True
        for (A, P, B, Q) in processes:
            min_cost = float('inf')
            # All S
            if A != 0:
                s = (mid + A - 1) // A
                cost = s * P
                if cost < min_cost:
                    min_cost = cost
            # All T
            if B != 0:
                t = (mid + B -1) // B
                cost = t * Q
                if cost < min_cost:
                    min_cost = cost
            # Iterate t from 0 to B+1
            for t in range(0, B + 2):
                needed = mid - t * B
                if needed <= 0:
                    s_needed = 0
                else:
                    s_needed = (needed + A -1) // A
                total = s_needed * A + t * B
                if total >= mid:
                    cost_candidate = s_needed * P + t * Q
                    if cost_candidate < min_cost:
                        min_cost = cost_candidate
            # Iterate s from 0 to A+1
            for s in range(0, A + 2):
                needed = mid - s * A
                if needed <= 0:
                    t_needed = 0
                else:
                    t_needed = (needed + B - 1) // B
                total = s * A + t_needed * B
                if total >= mid:
                    cost_candidate = s * P + t_needed * Q
                    if cost_candidate < min_cost:
                        min_cost = cost_candidate
            total_cost += min_cost
            if total_cost > X:
                feasible = False
                break
        if feasible:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    print(ans)

if __name__ == "__main__":
    main()