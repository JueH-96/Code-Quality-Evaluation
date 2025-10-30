import sys
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
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    queries = []
    ys = set()
    for _ in range(Q):
        x = int(input[ptr]); ptr +=1
        y = int(input[ptr]); ptr +=1
        queries.append((x, y))
        ys.add(y)
    ys.add(0)

    # Create sorted list S and map for coordinates
    S = sorted(ys)
    M = len(S)

    # Initialize Fenwick Trees
    count_tree = FenwickTree(M)
    sum_tree = FenwickTree(M)

    # Initialize all elements to 0
    zero_idx = bisect.bisect_left(S, 0)
    r_zero = zero_idx + 1
    count_tree.update(r_zero, N)
    sum_tree.update(r_zero, 0)

    # Track current values (1-based)
    current_values = [0] * (N + 2)  # 1..N are valid

    # Process each query
    for x, y in queries:
        old_val = current_values[x]
        new_val = y

        # Remove old_val from Fenwick Trees
        old_idx = bisect.bisect_left(S, old_val)
        old_r = old_idx + 1
        count_tree.update(old_r, -1)
        sum_tree.update(old_r, -old_val)

        # Add new_val to Fenwick Trees
        new_idx = bisect.bisect_left(S, new_val)
        new_r = new_idx + 1
        count_tree.update(new_r, 1)
        sum_tree.update(new_r, new_val)

        current_values[x] = new_val

        # Binary search to find v
        low = 0
        high = M - 1
        best_v = S[0]
        while low <= high:
            mid = (low + high) // 2
            v_candidate = S[mid]
            idx = bisect.bisect_left(S, v_candidate)
            r = idx + 1
            cnt_total = count_tree.query(M)
            current_count = cnt_total - count_tree.query(r - 1)
            if current_count >= K:
                best_v = v_candidate
                low = mid + 1
            else:
                high = mid - 1
        v = best_v

        # Compute sum_gt and count_gt
        i_plus = bisect.bisect_right(S, v)
        if i_plus >= M:
            sum_gt = 0
            count_gt = 0
        else:
            x_plus = S[i_plus]
            idx_plus = bisect.bisect_left(S, x_plus)
            r_plus = idx_plus + 1
            sum_total = sum_tree.query(M)
            sum_upto = sum_tree.query(r_plus - 1)
            sum_gt = sum_total - sum_upto

            count_total = count_tree.query(M)
            count_upto = count_tree.query(r_plus - 1)
            count_gt = count_total - count_upto

        total = sum_gt + v * (K - count_gt)
        print(total)

if __name__ == "__main__":
    main()