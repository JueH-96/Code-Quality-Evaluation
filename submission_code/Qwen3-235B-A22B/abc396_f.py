import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))

    class FenwickTree:
        def __init__(self, size):
            self.N = size + 2
            self.tree = [0] * (self.N)
        def update(self, idx, delta=1):
            while idx < self.N:
                self.tree[idx] += delta
                idx += idx & -idx
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    # Compute initial inversion count
    inv0 = 0
    ft = FenwickTree(M)
    for i in reversed(range(N)):
        inv0 += ft.query(A[i])
        ft.update(A[i] + 1)

    freq = [0] * M
    for a in A:
        freq[a] += 1

    diff = [0] * (M + 2)  # diff array for intervals

    for a in range(M):
        for b in range(M):
            if freq[a] == 0 or freq[b] == 0:
                continue
            cnt = freq[a] * freq[b]
            if a < b:
                l = (M - b) % M
                r = (M - a - 1) % M
                if l <= r:
                    diff[l] += cnt
                    diff[r + 1] -= cnt
                else:
                    diff[0] += cnt
                    diff[r + 1] -= cnt
                    diff[l] += cnt
                    diff[M] -= cnt
            elif a > b:
                l = (M - b) % M
                r = (M - a - 1) % M
                if l <= r:
                    diff[l] -= cnt
                    diff[r + 1] += cnt
                else:
                    diff[0] -= cnt
                    diff[r + 1] += cnt
                    diff[l] -= cnt
                    diff[M] += cnt

    # Compute prefix sums
    current = 0
    acc = [0] * M
    for i in range(M):
        current += diff[i]
        acc[i] = current

    # Build answers
    result = [(inv0 + acc[i]) for i in range(M)]
    # Adjust for modular wrap-around
    C = [0] * M
    for a in A:
        k = (M - 1 - a) % M
        C[k] += 1
    prefixC = [0] * (M + 1)
    for i in range(M):
        prefixC[i+1] = prefixC[i] + C[i]

    print('
'.join(map(str, result)))

if __name__ == "__main__":
    main()