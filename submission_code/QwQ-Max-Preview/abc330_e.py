class BIT:
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

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    max_num = N
    cnt = [0] * (max_num + 2)  # 0..N

    for a in A:
        if 0 <= a <= max_num:
            cnt[a] += 1

    bit = BIT(max_num + 1)
    for x in range(max_num + 1):
        if cnt[x] > 0:
            bit.update(x + 1, 1)

    for _ in range(Q):
        i = int(input[ptr]) - 1  # 0-based
        ptr += 1
        x = int(input[ptr])
        ptr += 1

        old_val = A[i]
        new_val = x
        A[i] = new_val

        # Process old_val
        if 0 <= old_val <= max_num:
            cnt[old_val] -= 1
            if cnt[old_val] == 0:
                bit.update(old_val + 1, -1)

        # Process new_val
        if 0 <= new_val <= max_num:
            cnt[new_val] += 1
            if cnt[new_val] == 1:
                bit.update(new_val + 1, 1)

        # Binary search for mex
        low = 0
        high = max_num
        res = max_num + 1
        while low <= high:
            mid = (low + high) // 2
            s = bit.query(mid + 1)
            if s < mid + 1:
                res = mid
                high = mid - 1
            else:
                low = mid + 1
        print(res)

if __name__ == "__main__":
    main()