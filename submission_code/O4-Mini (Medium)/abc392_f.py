import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:]))

    # Fenwick Tree / BIT for counts of empty slots
    size = n
    bit = [0] * (size + 1)

    # build initial: all slots empty => 1
    for i in range(1, size+1):
        bit[i] += 1
    # build tree
    for i in range(1, size+1):
        j = i + (i & -i)
        if j <= size:
            bit[j] += bit[i]

    def bit_add(i, v):
        # add v at position i
        while i <= size:
            bit[i] += v
            i += i & -i

    def bit_find_kth(k):
        # find smallest idx such that prefix sum >= k
        idx = 0
        # highest power of two <= size
        bit_mask = 1 << (size.bit_length() - 1)
        s = 0
        while bit_mask:
            t = idx + bit_mask
            if t <= size and s + bit[t] < k:
                s += bit[t]
                idx = t
            bit_mask >>= 1
        return idx + 1

    A = [0] * n

    # fill from i = n down to 1
    # P is 0-indexed: P[0] is P_1
    for i in range(n, 0, -1):
        k = P[i-1]
        pos = bit_find_kth(k)
        A[pos-1] = i
        bit_add(pos, -1)

    # output
    out = ' '.join(map(str, A))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()