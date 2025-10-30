import math
import sys

def get_square_root(num):
    if num < 0:
        return -1
    left = 0
    right = num
    while left <= right:
        mid = (left + right) // 2
        mid_sq = mid * mid
        if mid_sq == num:
            return mid
        elif mid_sq < num:
            left = mid + 1
        else:
            right = mid - 1
    return -1

data = sys.stdin.read().strip()
N = int(data)
sqrt_N = int(math.sqrt(N))
for i in range(1, sqrt_N + 1):
    if N % i == 0:
        d1 = i
        d2 = N // i
        # Check for d = d1
        m = N // d1
        D_disc = 3 * (4 * m - d1 * d1)
        if D_disc >= 0:
            s = get_square_root(D_disc)
            if s != -1:
                num = -3 * d1 + s
                if num % 6 == 0:
                    y_val = num // 6
                    if y_val > 0:
                        x_val = y_val + d1
                        print(x_val, y_val)
                        sys.exit()
        if d1 != d2:
            # Check for d = d2
            m2 = N // d2
            D_disc2 = 3 * (4 * m2 - d2 * d2)
            if D_disc2 >= 0:
                s2 = get_square_root(D_disc2)
                if s2 != -1:
                    num2 = -3 * d2 + s2
                    if num2 % 6 == 0:
                        y_val2 = num2 // 6
                        if y_val2 > 0:
                            x_val2 = y_val2 + d2
                            print(x_val2, y_val2)
                            sys.exit()
# No pair found
print(-1)
sys.exit()