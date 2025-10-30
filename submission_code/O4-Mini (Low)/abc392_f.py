import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:]))

    # Fenwick Tree / BIT for managing "empty slots"
    # We store in bit[i] the sum for the range (i - lsb(i) + 1) .. i
    bit = [0] * (n + 1)
    # Initialize with 1 at each position (all slots empty)
    for i in range(1, n + 1):
        bit[i] += 1
        j = i + (i & -i)
        if j <= n:
            bit[j] += bit[i]

    def bit_update(i, delta):
        while i <= n:
            bit[i] += delta
            i += i & -i

    def bit_find_kth(k):
        # find smallest pos such that prefix_sum(pos) >= k
        pos = 0
        # largest power of two <= n
        logn = 1 << (n.bit_length() - 1)
        step = logn
        while step:
            nxt = pos + step
            if nxt <= n and bit[nxt] < k:
                k -= bit[nxt]
                pos = nxt
            step >>= 1
        return pos + 1

    # answer array
    ans = [0] * (n + 1)

    # Process i = n..1, placing i into the P[i-1]-th empty slot
    for i in range(n, 0, -1):
        k = P[i - 1]
        pos = bit_find_kth(k)
        ans[pos] = i
        bit_update(pos, -1)

    # output
    out = sys.stdout
    out.write(" ".join(map(str, ans[1:])))
    out.write("
")

if __name__ == "__main__":
    main()