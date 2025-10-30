import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def add(self, idx, delta):
        # idx is 1-based
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        # returns sum from 1..idx (1-based)
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    A = list(map(int, input[idx:idx+N]))
    idx += N
    B_list = list(map(int, input[idx:idx+M]))
    idx += M

    ft = FenwickTree(N)

    for B in B_list:
        current_val = A[B] + ft.query(B + 1)
        delta_sub = -current_val
        L = B
        R = B
        ft.add(L + 1, delta_sub)
        ft.add(R + 2, -delta_sub)

        S = (B + 1) % N
        end = S + current_val - 1
        if end < N:
            l = S
            r = end
            ft.add(l + 1, 1)
            ft.add(r + 2, -1)
        else:
            l1 = S
            r1 = N - 1
            ft.add(l1 + 1, 1)
            ft.add(r1 + 2, -1)
            l2 = 0
            r2 = end % N
            ft.add(l2 + 1, 1)
            ft.add(r2 + 2, -1)

    res = []
    for i in range(N):
        val = A[i] + ft.query(i + 1)
        res.append(str(val))
    print(' '.join(res))

if __name__ == "__main__":
    main()