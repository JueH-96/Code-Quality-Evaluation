# YOUR CODE HERE
import math, sys

def main():
    import sys
    import math

    input = sys.stdin.read().split()
    N = int(input[0])
    P = list(map(int, input[1:N+1]))

    max_sum_A = [float('-inf')] * (N+1)
    max_sum_A[0] = 0.0

    for idx in range(N):
        p = P[idx]
        # Iterate k from current possible max down to 1
        for k in range(min(idx+1, N), 0, -1):
            candidate = max_sum_A[k-1] * 0.9 + p
            if candidate > max_sum_A[k]:
                max_sum_A[k] = candidate

    max_R = float('-inf')

    # Precompute powers of 0.9 to avoid recomputing
    pow_09 = [1.0] * (N+1)
    for k in range(1, N+1):
        pow_09[k] = pow_09[k-1] * 0.9

    for k in range(1, N+1):
        sum_B = (1 - pow_09[k]) / 0.1
        average = max_sum_A[k] / sum_B
        penalty = 1200 / math.sqrt(k)
        R = average - penalty
        if R > max_R:
            max_R = R

    print(max_R)

if __name__ == "__main__":
    main()