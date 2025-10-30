import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    
    processes = []
    for _ in range(N):
        A = int(input[ptr])
        ptr += 1
        P = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        Q = int(input[ptr])
        ptr += 1
        processes.append((A, P, B, Q))
    
    def min_cost(A, P, B, Q, C):
        if C == 0:
            return 0
        min_c = float('inf')
        # All S
        x_s = (C + A - 1) // A
        cost_s = x_s * P
        min_c = min(min_c, cost_s)
        # All T
        y_t = (C + B - 1) // B
        cost_t = y_t * Q
        min_c = min(min_c, cost_t)
        K = 100
        # Try x from 0 to K
        for x in range(0, K + 1):
            a_cont = x * A
            if a_cont >= C:
                cost = x * P
                if cost < min_c:
                    min_c = cost
                continue
            rem = C - a_cont
            y = (rem + B - 1) // B
            cost = x * P + y * Q
            if cost < min_c:
                min_c = cost
        # Try y from 0 to K
        for y in range(0, K + 1):
            b_cont = y * B
            if b_cont >= C:
                cost = y * Q
                if cost < min_c:
                    min_c = cost
                continue
            rem = C - b_cont
            x = (rem + A - 1) // A
            cost = x * P + y * Q
            if cost < min_c:
                min_c = cost
        return min_c
    
    low = 0
    high = 10**18
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        total = 0
        for (A, P, B, Q) in processes:
            c = min_cost(A, P, B, Q, mid)
            total += c
            if total > X:
                break
        if total <= X:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == "__main__":
    main()