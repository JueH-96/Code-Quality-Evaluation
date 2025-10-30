import sys
from math import log2, floor

def query(i, j):
    print(f"? {i} {j}", flush=True)
    return int(input())

def main():
    N, L, R = map(int, input().split())
    logL = floor(log2(L + 1))
    logR = floor(log2(R + 1))
    
    # Calculate the remainder for the entire range [0, 2^N - 1]
    total_sum_mod_100 = 0
    for i in range(N, -1, -1):
        j = 0
        while 2 ** i * (j + 1) <= 2 ** N:
            total_sum_mod_100 += query(i, j)
            j += 1
    
    # Calculate the remainder for the range [0, L - 1]
    sum_mod_100_before_L = 0
    for i in range(logL, -1, -1):
        j = L // (2 ** i)
        if j * (2 ** i) < L:
            j += 1
        if j > 0:
            sum_mod_100_before_L += query(i, j - 1)
    
    # Calculate the remainder for the range [R + 1, 2^N - 1]
    sum_mod_100_after_R = 0
    for i in range(logR, -1, -1):
        j = R // (2 ** i) + 1
        if j * (2 ** i) <= R:
            j += 1
        if j > 0:
            sum_mod_100_after_R += query(i, j - 1)
    
    # Calculate the final remainder
    final_sum_mod_100 = (total_sum_mod_100 - sum_mod_100_before_L - sum_mod_100_after_R) % 100
    
    print(f"! {final_sum_mod_100}", flush=True)

if __name__ == "__main__":
    main()