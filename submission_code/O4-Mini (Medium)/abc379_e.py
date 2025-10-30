def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    # P[i] = sum_{k=1..i} digit(S[k]) * k
    P = [0] * (N + 1)
    for i, ch in enumerate(S, start=1):
        # digit value times its index
        P[i] = P[i-1] + (ord(ch) - 48) * i

    # We'll build the decimal digits of the answer in little-endian in C.
    # The coefficient for 10^t is A[t] = P[N-t], for t=0..N-1.
    # Then we normalize carries base 10.
    # We allocate a little extra space for carries spilling beyond N.
    C = [0] * (N + 20)
    # Fill initial coefficients
    # C[0] = P[N], C[1] = P[N-1], ..., C[N-1] = P[1]
    # Beyond that, zeros.
    for t in range(N):
        C[t] = P[N - t]

    # Normalize carries
    carry = 0
    for i in range(len(C)-1):
        total = C[i] + carry
        # digit
        C[i] = total % 10
        # carry to next
        carry = total // 10
    # Last position
    C[-1] = carry

    # Find most significant non-zero digit
    msd = len(C) - 1
    while msd > 0 and C[msd] == 0:
        msd -= 1

    # Output digits from msd down to 0
    out = ''.join(str(C[i]) for i in range(msd, -1, -1))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()