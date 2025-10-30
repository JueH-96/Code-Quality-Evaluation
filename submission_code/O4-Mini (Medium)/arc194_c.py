import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    C = [int(next(it)) for _ in range(n)]

    # Initial total weight of ones
    W = 0
    # Lists of costs for removals (A=1->B=0) and insertions (A=0->B=1)
    R = []
    I = []
    for a, b, c in zip(A, B, C):
        if a == 1:
            W += c
            if b == 0:
                R.append(c)
        else:  # a == 0
            if b == 1:
                I.append(c)

    # Sort removals in descending order of C, so we remove big weights first
    R.sort(reverse=True)
    # Sort insertions in ascending order of C, so we insert small weights earlier
    I.sort()

    total_cost = 0
    cur = W

    # Perform all removals
    # After flipping a 1->0 with cost c, cur decreases by c, and we pay cur
    for c in R:
        cur -= c
        total_cost += cur

    # Perform all insertions
    # After flipping a 0->1 with cost c, cur increases by c, and we pay cur
    for c in I:
        cur += c
        total_cost += cur

    # Output the result
    print(total_cost)

if __name__ == "__main__":
    main()