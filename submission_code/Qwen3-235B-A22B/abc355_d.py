import bisect

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

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    intervals = []
    all_A = []
    all_B = []
    for _ in range(N):
        l = int(input[ptr])
        r = int(input[ptr + 1])
        intervals.append((l, r))
        all_A.append(r)
        all_A.append(l)
        all_B.append(l)
        all_B.append(r)
        ptr += 2

    # Process sorted_A
    sorted_A = sorted(list(set(all_A)))
    size_A = len(sorted_A)

    ft_A = FenwickTree(size_A)
    A = 0
    for l, r in intervals:
        x = l - 1
        query_pos = bisect.bisect_right(sorted_A, x)
        if query_pos > 0:
            a_count = ft_A.query(query_pos)
        else:
            a_count = 0
        A += a_count
        insert_pos = bisect.bisect_right(sorted_A, r)
        if insert_pos <= size_A:
            ft_A.update(insert_pos, 1)

    # Process sorted_B
    sorted_B = sorted(list(set(all_B)))
    size_B = len(sorted_B)

    ft_B = FenwickTree(size_B)
    B = 0
    current_size = 0
    for l, r in intervals:
        query_x = r
        query_pos = bisect.bisect_right(sorted_B, query_x)
        b_count = current_size - ft_B.query(query_pos)
        B += b_count
        insert_pos = bisect.bisect_right(sorted_B, l)
        if insert_pos <= size_B:
            ft_B.update(insert_pos, 1)
        current_size += 1

    T = N * (N - 1) // 2
    answer = T - (A + B)
    print(answer)

if __name__ == '__main__':
    main()