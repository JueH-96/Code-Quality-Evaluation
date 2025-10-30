# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.read().split()))
    
    D_diff = [0]*(N+2)
    C_diff = [0]*(N+2)
    total_cw =0
    for i in range(M-1):
        X1 = X[i]
        X2 = X[i+1]
        d_cw = (X2 - X1 + N) % N
        total_cw += d_cw
        if X1 < X2:
            D_diff[X1] += d_cw
            D_diff[X2] -= d_cw
            C_diff[X1] +=1
            C_diff[X2] -=1
        else:
            D_diff[X1] += d_cw
            D_diff[N+1] -= d_cw
            D_diff[1] += d_cw
            D_diff[X2] -= d_cw
            C_diff[X1] +=1
            C_diff[N+1] -=1
            C_diff[1] +=1
            C_diff[X2] -=1
    sum_dk = 0
    sum_ck =0
    min_sum_k = float('inf')
    current_d = 0
    current_c =0
    for k in range(1, N+1):
        current_d += D_diff[k]
        current_c += C_diff[k]
        sum_k = total_cw + N * current_c - 2 * current_d
        if sum_k < min_sum_k:
            min_sum_k = sum_k
    print(min_sum_k)

if __name__ == "__main__":
    main()