import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    # Map each value to its original position (1-based)
    pos_of = [0] * (N+1)
    for i, v in enumerate(P, start=1):
        pos_of[v] = i

    # Fenwick (BIT) for prefix sums / order statistics
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n+1)
        def update(self, i, v):
            # add v at index i
            while i <= self.n:
                self.fw[i] += v
                i += i & -i
        def query(self, i):
            # sum from 1..i
            s = 0
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s
        def range_sum(self, l, r):
            return self.query(r) - self.query(l-1)

    bit = Fenwick(N)
    # initially mark all positions present
    for i in range(1, N+1):
        bit.update(i, 1)

    total = 0
    # We "extract" values from N down to 1, placing each at its target end
    # The current index of value x among remaining is bit.query(orig_pos).
    # Its target index is x (since we will have x items once it's its turn).
    # If pos < target, it moves right across boundaries pos..target-1:
    #    cost = sum(pos..target-1) = (pos + (target-1))*(target-pos)//2
    # If pos > target, similarly moving left across boundaries target..pos-1:
    #    cost = sum(target..pos-1) = ((target) + (pos-1))*(pos-target)//2
    for x in range(N, 0, -1):
        orig = pos_of[x]
        pos = bit.query(orig)
        target = x
        if pos < target:
            d = target - pos
            # sum from pos to target-1
            cost = (pos + (target-1)) * d // 2
            total += cost
        elif pos > target:
            d = pos - target
            # sum from target to pos-1
            cost = (target + (pos-1)) * d // 2
            total += cost
        # remove x from remaining
        bit.update(orig, -1)

    print(total)


if __name__ == "__main__":
    main()