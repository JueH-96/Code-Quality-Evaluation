import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    processes = []
    for _ in range(N):
        A = int(next(it))
        P = int(next(it))
        B = int(next(it))
        Q = int(next(it))
        processes.append((A, P, B, Q))
    
    low = 0
    high = 10**18
    
    while low <= high:
        mid = (low + high) // 2
        total_cost = 0
        over_budget = False
        for (A, P, B_val, Q) in processes:
            cost_i = float('inf')
            for r in range(B_val):
                if mid <= r * A:
                    candidate = r * P
                    if candidate < cost_i:
                        cost_i = candidate
                else:
                    num = mid - r * A
                    Z = (num + B_val - 1) // B_val
                    c1 = Z * Q
                    k_low = Z // A
                    c2 = k_low * (B_val * P) + (Z - k_low * A) * Q
                    k_high = (Z + A - 1) // A
                    c3 = k_high * (B_val * P)
                    candidate_mix = min(c1, c2, c3)
                    total_candidate = r * P + candidate_mix
                    if total_candidate < cost_i:
                        cost_i = total_candidate
            total_cost += cost_i
            if total_cost > X:
                over_budget = True
                break
        if over_budget:
            high = mid - 1
        else:
            low = mid + 1
    print(high)

if __name__ == '__main__':
    main()