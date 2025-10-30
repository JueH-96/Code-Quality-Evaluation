def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # If there's only one pile, Fennec takes it and wins immediately.
    if N == 1:
        print("Fennec")
        return

    # If N is even, without any stalls the N-th activation is on an even move,
    # so Snuke wins.  Fennec can never force an odd total move‐count shift
    # against optimal Snuke, so Snuke wins for all even N.
    if N % 2 == 0:
        print("Snuke")
        return

    # Now N is odd and N >= 3.  In this case Fennec wins
    # as long as not *every* A_i is even.  If all A_i are even,
    # Snuke can use the symmetry of stalls to force a win.
    for a in A:
        if a & 1:
            print("Fennec")
            break
    else:
        print("Snuke")

if __name__ == "__main__":
    main()