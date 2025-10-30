import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    P = list(map(int, (next(it) for _ in range(n))))
    # Fenwick tree for counts of values 1..n
    size = n
    bit = [0] * (size + 1)
    def bit_add(i, v):
        # add v at index i (1-based)
        while i <= size:
            bit[i] += v
            i += i & -i
    def bit_sum(i):
        # sum of [1..i]
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    ans = 0
    # process insertion-sort cost
    # for i from 1 to n, let less_i = # of P[j] < P[i] for j < i
    # cost_i = (i-1)*i//2 - less_i*(less_i+1)//2
    for idx, x in enumerate(P, start=1):
        # count how many seen values < x
        less = bit_sum(x - 1)
        # total swaps cost for this element
        # sum_{k=less+1 to idx-1} k = sum_{1..idx-1} - sum_{1..less}
        # = (idx-1)*idx//2 - less*(less+1)//2
        cost_i = (idx - 1) * idx // 2 - (less * (less + 1) // 2)
        ans += cost_i
        # mark value x seen
        bit_add(x, 1)

    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()