import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

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
    all_values = [0]
    for _ in range(Q):
        X = int(input[ptr]); ptr +=1
        Y = int(input[ptr]); ptr +=1
        queries.append( (X, Y) )
        all_values.append(Y)

    # Coordinate compression
    sorted_unique = sorted( list( set(all_values) ) )
    m = len(sorted_unique)
    value_to_rank = {v:i for i, v in enumerate(sorted_unique)}

    # Initialize Fenwick Trees
    count_bit = FenwickTree(m)
    sum_bit = FenwickTree(m)

    # Find zero's rank
    zero_val = 0
    zero_rank = value_to_rank[zero_val]
    count_bit.update(zero_rank + 1, N)
    sum_bit.update(zero_rank + 1, 0)

    current_values = [0] * (N + 1)  # 1-based

    for X, Y in queries:
        old_val = current_values[X]
        # Process old_val
        old_rank = value_to_rank[old_val]
        count_bit.update(old_rank + 1, -1)
        sum_bit.update(old_rank + 1, -old_val)

        # Process new_val
        new_val = Y
        new_rank = value_to_rank[new_val]
        count_bit.update(new_rank + 1, 1)
        sum_bit.update(new_rank + 1, new_val)

        current_values[X] = new_val

        # Now perform binary search
        total_count = N
        sum_total = sum_bit.query(m)

        low = 0
        high = m - 1
        answer_mid = 0
        while low <= high:
            mid = (low + high) // 2
            count_lt = count_bit.query(mid)
            count_ge = total_count - count_lt
            if count_ge >= K:
                answer_mid = mid
                low = mid + 1
            else:
                high = mid - 1

        # Now calculate the sum
        current_val = sorted_unique[answer_mid]
        count_lt = count_bit.query(answer_mid)
        count_ge = total_count - count_lt

        # Compute count_gt
        count_le_current = count_bit.query(answer_mid + 1)
        count_gt = total_count - count_le_current

        sum_gt = sum_total - sum_bit.query(answer_mid + 1)
        remaining = K - count_gt
        sum_result = sum_gt + remaining * current_val

        print(sum_result)

if __name__ == "__main__":
    main()