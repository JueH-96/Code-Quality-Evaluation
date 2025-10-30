import sys
import math

def main():
    N_X = sys.stdin.readline().split()
    N = int(N_X[0])
    X = int(N_X[1])
    
    processes = []
    for _ in range(N):
        A_P_B_Q = sys.stdin.readline().split()
        A = int(A_P_B_Q[0])
        P = int(A_P_B_Q[1])
        B = int(A_P_B_Q[2])
        Q = int(A_P_B_Q[3])
        processes.append((A, P, B, Q))
    
    # Binary search on the production capacity
    low = 0
    high = 10**18  # Upper bound for binary search
    best_C = 0
    while low <= high:
        C = (low + high) // 2
        total_cost = 0
        for A, P, B, Q in processes:
            min_cost = float('inf')
            # Iterate over possible t (units of T)
            for t in range(B):
                remaining = C - t * B
                if remaining <= 0:
                    s = 0
                else:
                    s = math.ceil(remaining / A)
                cost = s * P + t * Q
                if cost < min_cost:
                    min_cost = cost
            total_cost += min_cost
        if total_cost <= X:
            best_C = C
            low = C + 1
        else:
            high = C - 1
    print(best_C)

if __name__ == "__main__":
    main()