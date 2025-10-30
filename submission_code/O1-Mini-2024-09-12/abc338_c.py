# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    N = int(sys.stdin.readline())
    Q = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Compute x_max
    x_max = float('inf')
    for ai, qi in zip(A, Q):
        if ai >0:
            x_max = min(x_max, qi // ai)
    if x_max == float('inf'):
        x_max = 0
    else:
        x_max = int(x_max)
    
    max_S =0
    # Precompute indices where B_i >0
    B_indices = [i for i in range(N) if B[i] >0]
    
    for x in range(0, x_max+1):
        y_max = float('inf')
        for i in B_indices:
            Q_left = Q[i] - A[i] * x
            if Q_left <0:
                y_max = -1
                break
            tmp = Q_left // B[i]
            if tmp < y_max:
                y_max = tmp
        if y_max <0:
            y_max =0
        S = x + y_max
        if S > max_S:
            max_S = S
    print(max_S)

if __name__ == "__main__":
    main()