MOD = 998244353
inv2 = pow(2, MOD - 2, MOD)

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def count_inversions(P):
    n = len(P)
    fenwick = FenwickTree(n)
    inv_count = 0
    for i in range(n):
        x = P[i]
        inv_count += i - fenwick.query(x)
        fenwick.update(x, 1)
    return inv_count

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N

    if N == 0 or K == 0:
        print(0)
        return

    D = N - K + 1
    if D == 0:
        print(0)
        return

    # Compute S1_mod
    s1_mod = (( ( (N - K + 1) % MOD ) * (K % MOD) % MOD ) * ( ( (K - 1) % MOD ) ) % MOD ) * inv2 % MOD

    # Compute S2_mod
    s2 = count_inversions(P)
    s2_mod = s2 % MOD

    # Compute S3_mod
    fenwick = FenwickTree(N)
    current_inv = 0
    for i in range(K):
        x = P[i]
        current_inv += fenwick.query(N) - fenwick.query(x)
        fenwick.update(x, 1)
    s3 = current_inv

    for i in range(1, D):
        x = P[i - 1]
        cnt_less = fenwick.query(x - 1)
        current_inv -= cnt_less
        fenwick.update(x, -1)

        y = P[i + K - 1]
        cnt_greater = fenwick.query(N) - fenwick.query(y)
        current_inv += cnt_greater
        fenwick.update(y, 1)

        s3 += current_inv

    s3_mod = s3 % MOD

    # Compute result
    inv_D = pow(D, MOD - 2, MOD)
    term1 = (s1_mod * inv2) % MOD
    term2 = ( ( (D % MOD) * s2_mod ) % MOD - s3_mod ) % MOD
    total = (term1 + term2) % MOD
    result = (total * inv_D) % MOD

    print(result)

if __name__ == '__main__':
    main()