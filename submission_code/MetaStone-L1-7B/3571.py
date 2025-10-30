class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)

    def update(self, index, value):
        index += 1  # 1-based indexing
        while index <= self.size:
            if self.tree[index] < value:
                self.tree[index] = value
            else:
                break
            index += index & -index

    def query(self, index):
        index += 1  # 1-based indexing
        res = 0
        while index > 0:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [0] * (2 * self.n)

    def update(self, pos, value):
        pos += self.n
        if self.tree[pos] >= value:
            return
        self.tree[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val

    def query(self, l, r):
        res = 0
        l += self.n
        r += self.n
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    k = int(input[idx])
    idx += 1
    points = []
    for i in range(n):
        x = int(input[idx])
        idx += 1
        y = int(input[idx])
        idx += 1
        points.append((x, y, i))

    points_sorted = sorted(points, key=lambda p: (p[0], p[1]))
    y_values = [p[1] for p in points_sorted]
    sorted_y = sorted(list(set(y_values)))
    rank = {y: i+1 for i, y in enumerate(sorted_y)}
    max_rank = len(sorted_y)

    # Forward pass
    ft_forward = SegmentTree(max_rank)
    forward = [0] * n
    for x, y, i in points_sorted:
        current_rank = rank[y]
        if current_rank == 1:
            q = 0
        else:
            q = ft_forward.query(1, current_rank - 1)
        forward[i] = q + 1
        ft_forward.update(current_rank, forward[i])

    # Backward pass
    points_sorted_back = sorted(points, key=lambda p: (-p[0], -p[1]))
    ft_backward = SegmentTree(max_rank)
    backward = [0] * n
    for x, y, i in points_sorted_back:
        current_rank = rank[y]
        if current_rank == max_rank:
            q = 0
        else:
            q = ft_backward.query(current_rank + 1, max_rank)
        backward[i] = q + 1
        ft_backward.update(current_rank, backward[i])

    # Compute the maximum path length that includes the k-th point
    max_length = 0
    for i in range(n):
        current_k = k
        current_rank = rank[points[current_k][1]]
        forward_val = forward[current_k]
        backward_val = backward[current_k]
        total = forward_val + backward_val - 1
        if total > max_length:
            max_length = total

    print(max_length)

if __name__ == '__main__':
    main()