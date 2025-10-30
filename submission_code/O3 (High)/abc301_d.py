def main() -> None:
    import sys

    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())

    L = len(S)                       # number of bits
    base = 0                         # value when every '?' is 0
    q_weights = []                   # powers of two for each '?' (MSB → LSB)

    for i, ch in enumerate(S):
        w = 1 << (L - 1 - i)         # weight of this bit
        if ch == '1':
            base |= w
        elif ch == '?':
            q_weights.append(w)      # remember, we will try to set these to 1

    # even the smallest possible value already exceeds N -> impossible
    if base > N:
        print(-1)
        return

    current = base
    # try to turn '?' bits to 1 from the most significant side
    for w in q_weights:              # q_weights is MSB→LSB as we scanned S
        if current + w <= N:
            current += w

    print(current)


if __name__ == "__main__":
    main()