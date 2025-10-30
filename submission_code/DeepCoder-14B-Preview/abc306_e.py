import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # Using n+2 to avoid issues with 1-based indexing

    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    updates = []
    y_values = set()
    y_values.add(0)  # initial state includes zero.

    for _ in range(Q):
        x = int(input[ptr]); ptr +=1
        y = int(input[ptr]); ptr +=1
        updates.append( (x, y) )
        y_values.add(y)

    # Create sorted list of unique values.
    sorted_values = sorted(y_values)
    M = len(sorted_values)

    # Initialize Fenwick Trees.
    count_tree = FenwickTree(M)
    sum_tree = FenwickTree(M)

    # Find the index of zero in sorted_values.
    zero_index = bisect.bisect_left(sorted_values, 0)
    if zero_index < M and sorted_values[zero_index] == 0:
        # Add N zeros initially.
        count_tree.update(zero_index + 1, N)  # because each element is zero.
        sum_tree.update(zero_index + 1, 0)  # sum is zero.

    # Initialize the array A.
    A = [0] * N

    for x, y in updates:
        idx = x - 1  # convert to 0-based index.

        old_val = A[idx]
        new_val = y

        # Remove old_val.
        old_index = bisect.bisect_left(sorted_values, old_val)
        if old_index < M and sorted_values[old_index] == old_val:
            count_tree.update(old_index + 1, -1)
            sum_tree.update(old_index + 1, -old_val)

        # Add new_val.
        new_index = bisect.bisect_left(sorted_values, new_val)
        if new_index < M and sorted_values[new_index] == new_val:
            count_tree.update(new_index + 1, 1)
            sum_tree.update(new_index + 1, new_val)
        else:
            # This should not happen as new_val is in sorted_values.
            pass

        # Update the array.
        A[idx] = new_val

        # Compute the sum of top K elements.

        # Binary search to find the best v.
        left = 0
        right = M - 1
        best = -1
        while left <= right:
            mid = (left + right) // 2
            v = sorted_values[mid]

            # Compute count_ge for v.
            index_lt = bisect.bisect_left(sorted_values, v)
            count_lt = count_tree.query(index_lt)
            count_ge = N - count_lt

            if count_ge >= K:
                best = mid
                left = mid + 1
            else:
                right = mid - 1

        if best == -1:
            # All elements are less than K, but since K <=N, this can't happen.
            # So we can set best=0.
            best = 0
        v = sorted_values[best]

        # Compute sum_gt and count_gt.
        index_lt = bisect.bisect_left(sorted_values, v)
        count_lt = count_tree.query(index_lt)

        index_gt = bisect.bisect_right(sorted_values, v)
        sum_le = sum_tree.query(index_gt)
        sum_gt = sum_tree.query(M) - sum_le

        count_le = count_tree.query(index_gt)
        count_gt = N - count_le

        remaining = K - count_gt
        if remaining < 0:
            remaining = 0
        sum_v = remaining * v

        total_sum = sum_gt + sum_v

        print(total_sum)

if __name__ == '__main__':
    main()