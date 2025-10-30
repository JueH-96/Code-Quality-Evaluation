import sys

def main():
    data = sys.stdin.read().split()
    if not data:
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
        for A, P, B, Q in processes:
            if A * Q < B * P:
                x_max = min(B - 1, (mid + A - 1) // A)
                best_cost = float('inf')
                for x in range(0, x_max + 1):
                    produced = x * A
                    if produced >= mid:
                        cost_here = x * P
                    else:
                        remaining = mid - produced
                        y = (remaining + B - 1) // B
                        cost_here = x * P + y * Q
                    if cost_here < best_cost:
                        best_cost = cost_here
            else:
                y_max = min(A - 1, (mid + B - 1) // B)
                best_cost = float('inf')
                for y in range(0, y_max + 1):
                    produced = y * B
                    if produced >= mid:
                        cost_here = y * Q
                    else:
                        remaining = mid - produced
                        x = (remaining + A - 1) // A
                        cost_here = x * P + y * Q
                    if cost_here < best_cost:
                        best_cost = cost_here
            total_cost += best_cost
            if total_cost > X:
                over_budget = True
                break
        
        if not over_budget and total_cost <= X:
            low = mid + 1
        else:
            high = mid - 1
    
    print(high)

if __name__ == '__main__':
    main()