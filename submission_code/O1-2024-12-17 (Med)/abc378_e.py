def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    # Fenwick (Binary Indexed) Tree for frequencies and for (value * frequency)
    class Fenwick:
        def __init__(self, size):
            self.size = size
            self.data = [0]*(size+1)
        def update(self, idx, val):
            # idx: 1-based index
            while idx <= self.size:
                self.data[idx] += val
                idx += idx & -idx
        def query(self, idx):
            # sum from 1..idx (inclusive)
            s = 0
            while idx > 0:
                s += self.data[idx]
                idx -= idx & -idx
            return s

    # Build prefix sums mod M
    P = [0]*(N+1)
    for i in range(N):
        P[i+1] = (P[i] + A[i]) % M

    # Fenwicks for counting prefix sums and summing their values
    freqFenw = Fenwick(M)   # to count frequencies of prefix sums mod M
    valFenw  = Fenwick(M)   # to sum up (index * frequency)

    # Initialize with P[0] = 0
    # P[0] = 0 => update position (0+1) in Fenwicks
    freqFenw.update(1, 1)  # one occurrence of 0
    # For valFenw, adding 0 adds nothing, so we can skip or do it harmlessly:
    # valFenw.update(1, 0)

    total = 0
    sumAll = 0  # sum of all prefix sums we've accounted for so far (P[0..r-1])

    # Process each r from 1..N
    for r in range(1, N+1):
        p = P[r]
        # 1) Count how many prefix sums <= p
        cLessEq = freqFenw.query(p+1)   # number of X_l <= p
        sLessEq = valFenw.query(p+1)    # sum of those X_l (which are <= p)
        # 2) S(r) = p*CountAll - sumAll + (CountAll - cLessEq)*M
        #    CountAll = r  (since l goes 0..r-1)
        CountAll = r
        S_r = p*CountAll - sumAll + (CountAll - cLessEq)*M
        total += S_r
        # 3) Now insert p into Fenwicks (prefix sums for next iteration)
        freqFenw.update(p+1, 1)
        valFenw.update(p+1, p)
        sumAll += p

    print(total)

# Do not forget to call main()
main()