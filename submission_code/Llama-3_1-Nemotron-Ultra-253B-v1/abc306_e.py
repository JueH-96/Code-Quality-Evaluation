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
    N = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    queries = []
    all_values = [0]
    for _ in range(Q):
        X = int(input[ptr])-1; ptr +=1
        Y = int(input[ptr]); ptr +=1
        queries.append( (X, Y) )
        all_values.append(Y)

    # Compress values
    sorted_unique = sorted(list(set(all_values)))
    value_to_idx = {v: i+1 for i, v in enumerate(sorted_unique)}  # 1-based
    M = len(sorted_unique)

    count_tree = FenwickTree(M)
    sum_tree = FenwickTree(M)

    # Initialize with N zeros
    initial_count = N
    idx_zero = value_to_idx[0]
    count_tree.update(idx_zero, initial_count)
    sum_tree.update(idx_zero, 0 * initial_count)

    current_values = [0] * N

    for X, Y in queries:
        old_val = current_values[X]
        new_val = Y
        current_values[X] = new_val

        # Remove old_val
        old_idx = value_to_idx[old_val]
        count_tree.update(old_idx, -1)
        sum_tree.update(old_idx, -old_val)

        # Add new_val
        new_idx = value_to_idx[new_val]
        count_tree.update(new_idx, 1)
        sum_tree.update(new_idx, new_val)

        # Binary search for T
        total_count = N
        total_sum = sum_tree.query(M)

        left = 0
        right = M -1
        best_T = 0
        best_count = 0

        while left <= right:
            mid = (left + right) // 2
            T_candidate = sorted_unique[mid]

            # Find prefix_idx for T_candidate -1
            prefix_pos = bisect.bisect_right(sorted_unique, T_candidate -1)
            prefix_idx = prefix_pos -1

            if prefix_idx >=0:
                prefix_count = count_tree.query(prefix_idx +1)
            else:
                prefix_count = 0

            current_count = total_count - prefix_count

            if current_count >= K:
                best_T = T_candidate
                best_count = current_count
                left = mid +1
            else:
                right = mid -1

        # Compute sum_ge and count_ge for best_T
        prefix_pos = bisect.bisect_right(sorted_unique, best_T -1)
        prefix_idx = prefix_pos -1

        if prefix_idx >=0:
            prefix_sum = sum_tree.query(prefix_idx +1)
            prefix_count = count_tree.query(prefix_idx +1)
        else:
            prefix_sum = 0
            prefix_count = 0

        sum_ge = total_sum - prefix_sum
        count_ge = total_count - prefix_count

        sum_top = sum_ge - (count_ge - K) * best_T

        print(sum_top)

if __name__ == '__main__':
    main()