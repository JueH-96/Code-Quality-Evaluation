def main():
    import sys
    from bisect import bisect_right

    data = sys.stdin.read().strip().split()
    N, M, P = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:]))

    # Sort side dishes for efficient binary search
    B.sort()

    # Create a prefix sum array for B
    prefixB = [0] * (M + 1)
    for i in range(M):
        prefixB[i + 1] = prefixB[i] + B[i]

    total_price = 0

    # For each main dish price a, find how many side dishes
    # have price b <= (P - a). For those, the set meal price is (a + b).
    # For the rest, it is P.
    for a in A:
        needed = P - a
        if needed < 0:
            # If a itself exceeds P, all sets cost P.
            total_price += M * P
        else:
            # Number of side dishes with price <= needed
            pos = bisect_right(B, needed)
            # Sum of the dishes that do not exceed needed
            sum_side_dishes = prefixB[pos]
            # For those pos dishes, total is a*pos + sum_side_dishes
            # For the remaining (M-pos) dishes, total is (M-pos)*P
            total_price += a * pos + sum_side_dishes + (M - pos) * P

    print(total_price)

# Do NOT forget to call main()
if __name__ == "__main__":
    main()