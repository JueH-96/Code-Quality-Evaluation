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

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

    def point_query(self, idx):
        return self.range_query(idx, idx)


def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    intervals_for_p = [[] for _ in range(N + 2)]  # 1-based
    
    for _ in range(M):
        L = int(data[idx])
        idx += 1
        R = int(data[idx])
        idx += 1
        X = int(data[idx])
        idx += 1
        intervals_for_p[X].append((L, R))
    
    MOD = 998244353
    bit = FenwickTree(N)
    result = 1
    
    for m in range(N, 0, -1):
        forbidden = 0
        valid_position = None
        for p in range(1, N + 1):
            if bit.point_query(p) == 1:
                continue
            is_forbidden_p = False
            for (L, R) in intervals_for_p[p]:
                sum_occ = bit.range_query(L, R)
                if sum_occ == 0:
                    is_forbidden_p = True
                    break
            if is_forbidden_p:
                forbidden += 1
            else:
                if valid_position is None:
                    valid_position = p
        valid = m - forbidden
        result = (result * valid) % MOD
        bit.update(valid_position, 1)
    
    print(result)


if __name__ == "__main__":
    main()