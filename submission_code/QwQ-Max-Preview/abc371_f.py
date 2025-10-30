import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    queries = []
    all_coords = X.copy()
    for _ in range(Q):
        T_i = int(input[ptr]) - 1  # 0-based
        G_i = int(input[ptr+1])
        queries.append((T_i, G_i))
        all_coords.append(G_i)
        ptr += 2

    # Coordinate compression
    sorted_coords = sorted(set(all_coords))
    coord_map = {x: i for i, x in enumerate(sorted_coords)}
    max_coord = len(sorted_coords)

    # Segment tree implementation
    class SegmentTree:
        def __init__(self, size):
            self.n = 1
            while self.n < size:
                self.n <<= 1
            self.size = self.n
            self.tree_count = [0] * (2 * self.n)
            self.tree_sum = [0] * (2 * self.n)

        def update(self, idx, val_count, val_sum):
            idx += self.n
            self.tree_count[idx] += val_count
            self.tree_sum[idx] += val_sum
            idx >>= 1
            while idx >= 1:
                self.tree_count[idx] = self.tree_count[2*idx] + self.tree_count[2*idx+1]
                self.tree_sum[idx] = self.tree_sum[2*idx] + self.tree_sum[2*idx+1]
                idx >>= 1

        def query_count_sum(self, l, r):
            res_count = 0
            res_sum = 0
            l += self.n
            r += self.n
            while l <= r:
                if l % 2 == 1:
                    res_count += self.tree_count[l]
                    res_sum += self.tree_sum[l]
                    l += 1
                if r % 2 == 0:
                    res_count += self.tree_count[r]
                    res_sum += self.tree_sum[r]
                    r -= 1
                l >>= 1
                r >>= 1
            return res_count, res_sum

    st = SegmentTree(max_coord + 2)
    for x in X:
        idx = bisect.bisect_left(sorted_coords, x)
        st.update(idx, 1, x)

    current_positions = X.copy()
    current_positions.sort()
    total = 0

    for T_i, G_i in queries:
        x = current_positions[T_i]
        if x == G_i:
            continue
        x_idx = bisect.bisect_left(sorted_coords, x)
        G_idx = bisect.bisect_left(sorted_coords, G_i)
        if G_i > x:
            left = x_idx + 1
            right = G_idx - 1
            count, sum_p = st.query_count_sum(left, right)
            steps = (G_i - x) + (G_i + 1) * count - sum_p
            total += steps
            # Remove x and the interval [left, right]
            st.update(x_idx, -1, -x)
            if count > 0:
                st.update(left, -count, -sum_p)
            # Add G_i and G_i+1 * count
            st.update(G_idx, 1, G_i)
            new_p = G_i + 1
            new_p_idx = bisect.bisect_left(sorted_coords, new_p)
            if new_p_idx < len(sorted_coords) and sorted_coords[new_p_idx] == new_p:
                st.update(new_p_idx, count, new_p * count)
            else:
                # Handle dynamically (though this case is covered by all_coords)
                pass
        else:
            left = G_idx + 1
            right = x_idx - 1
            count, sum_p = st.query_count_sum(left, right)
            steps = (x - G_i) + (sum_p - (G_i + 1) * count)
            total += steps
            # Remove x and the interval [left, right]
            st.update(x_idx, -1, -x)
            if count > 0:
                st.update(left, -count, -sum_p)
            # Add G_i and G_i-1 * count
            st.update(G_idx, 1, G_i)
            new_p = G_i - 1
            new_p_idx = bisect.bisect_left(sorted_coords, new_p)
            if new_p_idx < len(sorted_coords) and sorted_coords[new_p_idx] == new_p:
                st.update(new_p_idx, count, new_p * count)
            else:
                # Handle dynamically (though this case is covered by all_coords)
                pass
    print(total)

if __name__ == '__main__':
    main()