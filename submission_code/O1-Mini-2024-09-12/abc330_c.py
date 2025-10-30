import math, sys

def main():
    D = int(sys.stdin.read())
    min_diff = float('inf')
    max_x = math.isqrt(D) + 1
    for x in range(0, max_x +1):
        x_sq = x * x
        if x_sq > D:
            y_sq = 0
            y = 0
            diff = abs(x_sq - D)
            if diff < min_diff:
                min_diff = diff
            continue
        rem = D - x_sq
        y = math.isqrt(rem)
        for candidate_y in [y, y+1]:
            if candidate_y <0:
                continue
            total = x_sq + candidate_y * candidate_y
            diff = abs(total - D)
            if diff < min_diff:
                min_diff = diff
                if min_diff ==0:
                    print(0)
                    return
    print(min_diff)

if __name__ == "__main__":
    main()