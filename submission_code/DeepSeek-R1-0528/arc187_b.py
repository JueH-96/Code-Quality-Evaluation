mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    B = list(map(lambda x: int(x) if x != '-1' else -1, data[2:2+n]))
    
    total_unknowns = sum(1 for x in B if x == -1)
    
    powM = [1] * (total_unknowns + 1)
    for i in range(1, total_unknowns + 1):
        powM[i] = powM[i-1] * m % mod

    min_fixed_arr = [0] * (n + 1)
    k_arr = [0] * (n + 1)
    min_fixed = m + 1
    count_unknown = 0
    min_fixed_arr[1] = m + 1
    k_arr[1] = 0

    for i in range(0, n - 1):
        if B[i] == -1:
            count_unknown += 1
        else:
            if min_fixed > B[i]:
                min_fixed = B[i]
        min_fixed_arr[i + 2] = min_fixed
        k_arr[i + 2] = count_unknown

    ans = powM[total_unknowns]
    
    for i in range(2, n + 1):
        if B[i - 1] != -1:
            x0 = B[i - 1]
            if x0 < min_fixed_arr[i]:
                k = k_arr[i]
                count_prefix = pow(m - x0, k, mod)
                count_suffix = powM[total_unknowns - k]
                F_i = count_prefix * count_suffix % mod
            else:
                F_i = 0
        else:
            k = k_arr[i]
            low = 1
            high = min_fixed_arr[i] - 1
            if high < low:
                F_i = 0
            else:
                s = 0
                if k == 0:
                    s = high - low + 1
                else:
                    for x0 in range(low, high + 1):
                        s = (s + pow(m - x0, k, mod)) % mod
                count_suffix = powM[total_unknowns - k - 1]
                F_i = s * count_suffix % mod
        ans = (ans + F_i) % mod
        
    print(ans % mod)

if __name__ == "__main__":
    main()