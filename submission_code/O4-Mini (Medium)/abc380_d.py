import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]
    L = len(S)
    Q = int(data[1])
    Ks = data[2:]
    # Ensure we have exactly Q queries
    # (If input is well-formed, len(Ks) == Q.)
    out = []
    # Using Python 3.8+ int.bit_count for fast popcount
    for i in range(Q):
        k = int(Ks[i])
        # zero‐based position
        pos = k - 1
        block = pos // L
        idx = pos - block * L  # same as pos % L
        flips = block.bit_count()
        c = S[idx]
        # If flips is odd, invert case
        if flips & 1:
            # swap case
            # faster than c.swapcase() in a tight loop may be direct
            if 'a' <= c <= 'z':
                c = c.upper()
            else:
                c = c.lower()
        out.append(c)
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()