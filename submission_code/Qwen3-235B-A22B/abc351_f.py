import bisect

class BIT:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing up to self.n

    def update(self, idx, val):
        # idx is 1-based
        while idx <= self.n:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        # returns sum from 1 to idx (1-based)
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))

    # Create S as sorted list of unique elements in A
    S = sorted(set(A))
    m = len(S)

    sum_bit = BIT(m)
    count_bit = BIT(m)
    total = 0

    for x in A:
        r = bisect.bisect_left(S, x)
        bit_idx = r + 1  # Convert to 1-based index for BIT

        sum_val = sum_bit.query(r)
        cnt_val = count_bit.query(r)
        total += x * cnt_val - sum_val

        sum_bit.update(bit_idx, x)
        count_bit.update(bit_idx, 1)

    print(total)

if __name__ == '__main__':
    main()