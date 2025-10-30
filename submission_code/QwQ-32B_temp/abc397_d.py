import math

def find_max_d(N):
    target = 4 * N
    left = 1
    right = 2 * 10**18  # A sufficiently large upper bound
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        mid_cubed = mid ** 3
        if mid_cubed <= target:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    return ans

def main():
    N = int(input().strip())
    max_d = find_max_d(N)
    found = False
    for d in range(1, max_d + 1):
        if N % d != 0:
            continue
        K = N // d
        D_val = 12 * K - 3 * d * d
        if D_val < 0:
            continue
        sqrt_D = math.isqrt(D_val)
        if sqrt_D * sqrt_D != D_val:
            continue
        numerator = (-3 * d) + sqrt_D
        if numerator <= 0:
            continue
        if numerator % 6 != 0:
            continue
        y = numerator // 6
        if y <= 0:
            continue
        x = y + d
        if x**3 - y**3 == N:
            print(x, y)
            found = True
            break
    if not found:
        print(-1)

if __name__ == "__main__":
    main()