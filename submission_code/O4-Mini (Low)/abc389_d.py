import sys
import threading

def main():
    import sys
    from math import isqrt

    data = sys.stdin.read().strip()
    if not data:
        return
    R = int(data)
    R2 = R * R
    # maximum i >= 0 with (i+0.5) <= R  => i <= R - 0.5
    i_max = int(R - 0.5)
    ans = 0
    # Precompute constant 4*R2
    fourR2 = 4 * R2
    for i in range(i_max + 1):
        # Compute fourS = 4*(R^2 - (i+0.5)^2) = 4*R2 - 4*i*(i+1) - 1
        fourS = fourR2 - 4 * i * (i + 1) - 1
        if fourS < 1:
            # No non-negative j satisfies (j+0.5)^2 <= R^2 - (i+0.5)^2
            continue
        # Maximum M = floor(sqrt(4S))
        M = isqrt(fourS)
        # 2*j + 1 <= M  => j <= (M-1)//2
        j_max = (M - 1) // 2
        if j_max < 0:
            continue

        if i == 0:
            # i = 0 row: for j=0 count 1, for j=1..j_max each gives 2 (±j)
            # total = 1 + 2*j_max
            ans += 1 + 2 * j_max
        else:
            # i > 0 rows: for j=0 gives 2 (±i,0), for j>=1 each quadrant gives 4
            # total = 2 + 4*j_max
            ans += 2 + 4 * j_max

    print(ans)

if __name__ == "__main__":
    main()