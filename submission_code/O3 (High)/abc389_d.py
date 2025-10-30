import sys, math

def main():
    R = int(sys.stdin.readline())
    R2 = R * R                      # R squared, used many times
    ans = 0

    # i runs from 0 up to floor(R-0.5) which equals R-1 for integer R ≥ 1
    for i in range(R):              # i = 0,1,…,R-1
        M = 2 * i + 1               # 2i + 1   (always odd)
        d = 4 * R2 - M * M          # (= 4R² - (2i+1)²)  always ≥ 1 in this loop
        smax = math.isqrt(d)        # floor(sqrt(d))
        N_i = (smax + 1) // 2       # number of non-negative j satisfying the inequality

        if i == 0:
            # (0,0) contributes 1; the other N_i-1 pairs with j>0 each contribute 2
            ans += 1 + (N_i - 1) * 2
        else:
            # pair (±i,0) contributes 2; each of the N_i-1 pairs with j>0 contributes 4
            ans += 2 + (N_i - 1) * 4

    print(ans)

if __name__ == "__main__":
    main()