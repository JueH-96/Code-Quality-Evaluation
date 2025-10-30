import math

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
        idx += 4

    low = 0
    high = 10**18
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        total = 0
        feasible = True
        for (A, P, B, Q) in processes:
            if mid == 0:
                cost = 0
            else:
                s_needed = (mid + A - 1) // A
                cost_s = s_needed * P

                t_needed = (mid + B - 1) // B
                cost_t = t_needed * Q

                s_max = (mid + A - 1) // A
                s_real = (mid * Q) / (A * Q + B * P)
                s_candidates = [
                    math.floor(s_real), math.ceil(s_real),
                    math.floor(s_real) - 1, math.ceil(s_real) + 1,
                    math.floor(s_real) - 2, math.ceil(s_real) + 2,
                    0, s_max - 1
                ]
                min_comb = float('inf')
                for s in s_candidates:
                    if s < 0 or s >= s_max:
                        continue
                    required = mid - A * s
                    if required <= 0:
                        continue
                    t = (required + B - 1) // B
                    current_cost = s * P + t * Q
                    if current_cost < min_comb:
                        min_comb = current_cost
                if min_comb == float('inf'):
                    min_comb = float('inf')
                cost = min(cost_s, cost_t, min_comb)
            total += cost
            if total > X:
                feasible = False
                break
        if feasible:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    print(ans)

if __name__ == '__main__':
    main()