import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    intervals = []
    idx = 1
    for i in range(N):
        l = int(data[idx])
        r = int(data[idx+1])
        intervals.append((l, r))
        idx += 2

    # Extract all r_i's for coordinate compression
    r_values = [r for (l, r) in intervals]
    sorted_r = sorted(r_values)
    unique_r = sorted(list(set(sorted_r)))
    rank_r = {v: i + 1 for i, v in enumerate(unique_r)}  # 1-based indexing

    # Sort intervals by their starting point
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # Binary Indexed Tree (Fenwick Tree) implementation
    class BIT:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (self.size + 2)  # 1-based indexing

        def update(self, idx, delta=1):
            while idx <= self.size:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    # Determine the maximum rank for the BIT
    max_rank = len(unique_r)
    bit = BIT(max_rank)

    total = 0

    # Process each interval in the sorted order
    for l, r in sorted_intervals:
        # Find the number of intervals with r_i > l
        # This is equivalent to finding the number of intervals where r_i > l
        # Using binary search on the sorted_r list
        # Find the first index where r > l
        idx = bisect.bisect_right(sorted_r, l)
        count = idx  # Number of elements > l is len(sorted_r) - idx
        # But wait, no. We need the number of elements in the BIT where r_i > l
        # Since the BIT contains all r_i's processed so far, which are the ones before current in the sorted list
        # Wait, no. The BIT is built as we process the intervals in the sorted order
        # So for the current interval, all previous intervals have been inserted
        # So the number of elements in the BIT where r_i > l is the number of elements with r_i > l
        # We can find this by querying the BIT for the sum from (rank of l + 1) to max_rank
        # Wait, no. The rank is based on the unique_r list, which may not include all possible r_i's
        # So perhaps the correct way is to find the number of elements in the BIT where r_i > l
        # Since the BIT contains all the r_i's processed so far, which are the ones before the current interval in the sorted list
        # Wait, no. The current interval is being processed, and the BIT contains all previous intervals
        # So to find the number of previous intervals where r_i > l, we can:
        # 1. Find the smallest r_i in unique_r that is > l
        # 2. The number of elements in the BIT with r_i > l is the total elements inserted so far minus the number of elements with r_i <= l
        # The total elements inserted so far is the number of elements processed before the current one, which is the number of elements in the BIT
        # Wait, no. The BIT's query function returns the sum up to a certain index, which represents the count of elements with rank <= that index
        # So the number of elements with r_i > l is:
        # total_elements_in_BIT - number_of_elements_with_r_i_leq_l
        # total_elements_in_BIT is the sum of all updates, which can be tracked separately
        # Alternatively, since we process intervals in order, the number of elements inserted so far is the number of intervals processed before the current one
        # Wait, no. The BIT's query function can give the total number of elements inserted so far if we query the maximum rank
        total_elements = bit.query(max_rank)
        # Find the number of elements with r_i <= l
        # Find the largest r in unique_r that is <= l
        pos = bisect.bisect_right(unique_r, l)
        count_le = bit.query(pos)  # since unique_r is sorted, pos is the number of elements <= l
        # The number of elements with r_i > l is total_elements - count_le
        current_count = total_elements - count_le
        total += current_count
        # Insert the current r into the BIT
        r_rank = rank_r[r]
        bit.update(r_rank)

    print(total)

if __name__ == '__main__':
    main()