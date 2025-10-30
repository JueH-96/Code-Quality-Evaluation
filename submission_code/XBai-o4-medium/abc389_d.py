import math

def main():
    import sys
    R = int(sys.stdin.readline())
    R_sq = R * R
    total = 0
    a = 0
    while True:
        a_half = a + 0.5
        a_half_sq = a_half * a_half
        if a_half_sq > R_sq:
            break
        remaining = R_sq - a_half_sq
        # Binary search for max b
        low = 0
        high = int(math.sqrt(remaining)) + 2
        best_b = 0
        while low <= high:
            mid = (low + high) // 2
            current = (mid + 0.5) ** 2
            if current <= remaining:
                best_b = mid
                low = mid + 1
            else:
                high = mid - 1
        b_max = best_b
        if a == 0:
            contribution = 1 + 2 * b_max
        else:
            contribution = 2 + 4 * b_max
        total += contribution
        a += 1
    print(total)

if __name__ == '__main__':
    main()