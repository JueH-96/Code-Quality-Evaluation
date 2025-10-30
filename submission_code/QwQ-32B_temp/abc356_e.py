import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

    def range_query(self, a, b):
        return self.query(b) - self.query(a - 1)

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A.sort()
    if not A:
        print(0)
        return
    max_A = A[-1]
    ft = FenwickTree(max_A)
    freq = [0] * (max_A + 2)  # 0-based for easy access
    total = 0

    for j in range(N):
        X = A[j]
        if j == 0:
            ft.update(X, 1)
            freq[X] += 1
            continue

        sqrt_X = int(X ** 0.5)
        sum1 = 0
        for Y in range(1, sqrt_X + 1):
            sum1 += (X // Y) * freq[Y]

        sum2 = 0
        for q in range(1, sqrt_X + 1):
            lower = (X // (q + 1)) + 1
            lower = max(lower, sqrt_X + 1)
            upper = X // q
            if lower > upper:
                continue
            if upper > max_A:
                upper = max_A
            cnt = ft.range_query(lower, upper)
            sum2 += q * cnt

        total += sum1 + sum2

        ft.update(X, 1)
        freq[X] += 1

    print(total)

if __name__ == "__main__":
    main()