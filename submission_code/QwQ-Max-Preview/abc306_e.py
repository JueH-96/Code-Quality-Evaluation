import bisect

class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.size:
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
    K = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    Xs = []
    Ys = []
    all_ys = set()
    for _ in range(Q):
        X = int(input[ptr])
        ptr += 1
        Y = int(input[ptr])
        ptr += 1
        Xs.append(X)
        Ys.append(Y)
        all_ys.add(Y)
    all_ys.add(0)
    compressed = sorted(all_ys)
    m = len(compressed)

    # Initialize BITs
    count_bit = BIT(m)
    sum_bit = BIT(m)

    # Initial setup: all elements are 0
    zero_val = 0
    zero_idx = bisect.bisect_left(compressed, zero_val)
    zero_bit_idx = zero_idx + 1  # 1-based
    count_bit.update(zero_bit_idx, N)
    sum_bit.update(zero_bit_idx, zero_val * N)

    curr_values = [0] * (N + 1)  # 1-based indexing for positions

    for i in range(Q):
        X = Xs[i]
        Y = Ys[i]
        old_val = curr_values[X]
        if old_val != Y:
            # Remove old_val from BITs
            old_idx = bisect.bisect_left(compressed, old_val)
            old_bit_idx = old_idx + 1
            count_bit.update(old_bit_idx, -1)
            sum_bit.update(old_bit_idx, -old_val)

            # Add new_val to BITs
            new_idx = bisect.bisect_left(compressed, Y)
            new_bit_idx = new_idx + 1
            count_bit.update(new_bit_idx, 1)
            sum_bit.update(new_bit_idx, Y)

            curr_values[X] = Y

        # Binary search to find T_index
        low = 0
        high = m - 1
        T_index = -1
        while low <= high:
            mid = (low + high) // 2
            # Compute count_ge for compressed[mid]
            bit_idx = mid + 1
            total_count = count_bit.query(m)
            if bit_idx == 1:
                current_count = total_count
            else:
                current_count = total_count - count_bit.query(bit_idx - 1)
            if current_count >= K:
                T_index = mid
                low = mid + 1
            else:
                high = mid - 1

        if T_index == -1:
            print(0)
        else:
            T = compressed[T_index]
            # Compute sum_ge(T_index)
            bit_idx = T_index + 1
            total_sum = sum_bit.query(m)
            if bit_idx == 1:
                sum_total = total_sum
            else:
                sum_total = total_sum - sum_bit.query(bit_idx - 1)
            # Compute count_ge again (already have current_count from search)
            # But current_count may be outdated, recompute to avoid errors
            # Alternatively, use the same method
            current_count = count_bit.query(m)
            if bit_idx != 1:
                current_count -= count_bit.query(bit_idx - 1)
            excess = current_count - K
            sum_ans = sum_total - excess * T
            print(sum_ans)

if __name__ == '__main__':
    main()