import math

def main():
    D = int(input())
    x_max = math.isqrt(D)
    min_diff = float('inf')
    
    for x in range(x_max + 1):
        rem = D - x * x
        if rem < 0:
            continue
        y_floor = math.isqrt(rem)
        candidates = [y_floor, y_floor + 1]
        for y in candidates:
            current_sum = x * x + y * y
            diff = abs(current_sum - D)
            if diff < min_diff:
                min_diff = diff
                if min_diff == 0:
                    print(0)
                    return
    print(min_diff)

if __name__ == "__main__":
    main()