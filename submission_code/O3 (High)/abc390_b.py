def main() -> None:
    import sys

    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:
        return

    N = data[0]
    A = data[1:]

    # With only two terms every sequence is a geometric progression
    if N <= 2:
        print("Yes")
        return

    is_gp = True
    for i in range(N - 2):
        # A_i * A_{i+2} must equal (A_{i+1})^2 in a GP
        if A[i] * A[i + 2] != A[i + 1] * A[i + 1]:
            is_gp = False
            break

    print("Yes" if is_gp else "No")


if __name__ == "__main__":
    main()