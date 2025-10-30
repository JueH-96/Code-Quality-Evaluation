import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    s = data[1]
    # S[i] = sum_{k=1}^i (digit_k * k)
    S = [0] * (N + 1)
    for i, ch in enumerate(s, start=1):
        S[i] = S[i-1] + (ord(ch) - 48) * i

    # We will build the answer in an array a of "digits" (but they may be > 9
    # initially).  The formula shows that the coefficient of 10^p in the answer
    # is sum_{k=1..N-p} (digit_k * k) = S[N-p].
    # So set a[p] = S[N-p] for p=0..N-1, then propagate carries.
    L = N + 20  # some extra space for carries
    a = [0] * L
    for p in range(N):
        a[p] = S[N - p]

    # Propagate carries in base 10
    for p in range(L - 1):
        carry = a[p] // 10
        a[p] %= 10
        a[p+1] += carry

    # Find highest non-zero digit
    i = L - 1
    while i > 0 and a[i] == 0:
        i -= 1

    # Output digits from most significant to least
    out = ''.join(str(a[j]) for j in range(i, -1, -1))
    sys.stdout.write(out)

if __name__ == '__main__':
    main()