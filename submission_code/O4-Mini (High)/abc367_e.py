import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    # read X and convert to 0-based
    p = [int(next(it)) - 1 for _ in range(n)]
    # read A
    A = [int(next(it)) for _ in range(n)]
    # pos[i] will track f^k(i)
    pos = list(range(n))
    # doubling method: p is currently f^(2^j) after j-th iteration
    # we consume bits of k from low to high
    while k > 0:
        if (k & 1):
            # apply current mapping to pos
            pos = [p[x] for x in pos]
        # square the mapping: p = p ◦ p
        p = [p[p_val] for p_val in p]
        k >>= 1

    # build the final array
    out = [str(A[pos_i]) for pos_i in pos]
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()