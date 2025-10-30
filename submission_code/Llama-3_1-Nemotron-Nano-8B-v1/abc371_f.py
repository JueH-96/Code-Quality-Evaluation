def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    X = list(map(int, input[ptr:ptr + N]))
    ptr += N

    Q = int(input[ptr])
    ptr += 1

    tasks = []
    all_pos = set()
    for _ in range(Q):
        T = int(input[ptr])
        G = int(input[ptr + 1])
        tasks.append((T, G))
        all_pos.add(G)
        ptr += 2
    all_pos.update(X)
    all_pos = sorted(all_pos)
    pos_map = {pos: i + 1 for i, pos in enumerate(all_pos)}
    M = len(all_pos)

    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)

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

    count_tree = FenwickTree(M)
    sum_tree = FenwickTree(M)

    pos = X.copy()
    for p in pos:
        idx = pos_map[p]
        count_tree.update(idx, 1)
        sum_tree.update(idx, p)

    total = 0

    for T, G in tasks:
        T -= 1
        current = pos[T]
        if current == G:
            continue

        x = current
        y = G

        x_coord = pos_map[x]
        y_coord = pos_map[y]

        if y > x:
            low = x_coord + 1
            high = y_coord - 1
            k = count_tree.query(high) - count_tree.query(low - 1)
            sum_p = sum_tree.query(high) - sum_tree.query(low - 1)
            steps = (y - x) + ((y + 1) * k - sum_p)
        else:
            low = y_coord + 1
            high = x_coord - 1
            k = count_tree.query(high) - count_tree.query(low - 1)
            sum_p = sum_tree.query(high) - sum_tree.query(low - 1)
            steps = (x - y) + (sum_p - (y - 1) * k)

        total += steps

        count_tree.update(x_coord, -1)
        sum_tree.update(x_coord, -x)
        count_tree.update(y_coord, 1)
        sum_tree.update(y_coord, y)
        pos[T] = y

    print(total)

if __name__ == '__main__':
    main()