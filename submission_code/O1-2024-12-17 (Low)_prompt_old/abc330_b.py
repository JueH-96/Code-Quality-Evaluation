def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data[:3])
    A = list(map(int, data[3:]))

    # Compute X_i by clamping A_i to [L, R].
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)

    print(" ".join(map(str, result)))

def main():
    solve()

if __name__ == "__main__":
    main()