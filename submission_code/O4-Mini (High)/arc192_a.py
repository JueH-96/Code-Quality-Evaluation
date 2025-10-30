def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [0]* (N+1)
    zeros = 0
    for i in range(1, N+1):
        ai = int(next(it))
        A[i] = ai
        if ai == 0:
            zeros += 1

    # No zeros: already good
    if zeros == 0:
        print("Yes")
        return

    # N odd
    if N & 1:
        # if all zeros, impossible; otherwise possible
        if zeros == N:
            print("No")
        else:
            print("Yes")
        return

    # N even
    # If N divisible by 4: always possible
    if N % 4 == 0:
        print("Yes")
        return

    # N mod 4 == 2: check parity-class-zero condition
    # Gather whether all even-indexed positions are zero,
    # or all odd-indexed positions are zero.
    all_even_zero = True
    all_odd_zero = True
    # positions are 1-indexed
    for i in range(1, N+1):
        if (i & 1) == 0:  # even position
            if A[i] != 0:
                all_even_zero = False
        else:  # odd position
            if A[i] != 0:
                all_odd_zero = False
        # early break
        if not all_even_zero and not all_odd_zero:
            break

    if all_even_zero or all_odd_zero:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()