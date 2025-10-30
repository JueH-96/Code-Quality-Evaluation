def main():
    import sys
    from functools import cmp_to_key

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    arr = []
    idx = 1

    # Read input and store (A, B, index)
    for i in range(1, N + 1):
        A = int(data[idx])
        B = int(data[idx + 1])
        idx += 2
        arr.append((A, B, i))

    # Custom comparison for sorting
    def compare(x, y):
        Ax, Bx, ix = x
        Ay, By, iy = y
        # Compare success rates Ax/(Ax+Bx) and Ay/(Ay+By) by cross multiplication
        lhs = Ax * (Ay + By)
        rhs = Ay * (Ax + Bx)
        if lhs == rhs:
            # Tie: compare by ascending index
            return ix - iy
        # Descending order by success rate
        return -1 if lhs > rhs else 1

    # Sort using the custom comparator
    arr.sort(key=cmp_to_key(compare))

    # Output the sorted indices
    print(" ".join(str(item[2]) for item in arr))

# Do not forget to call main.
main()