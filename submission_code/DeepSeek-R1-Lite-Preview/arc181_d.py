def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    M = int(data[N+1])
    A = list(map(int, data[N+2:N+2+M]))

    # Assign ranks
    rank = [0] * (N + 1)
    sorted_P = sorted(P)
    for i in range(N):
        rank[P[i]] = i + 1

    # Fenwick Tree (BIT) for inversion count
    class BIT:
        def __init__(self, size):
            self.N = size
            self.tree = [0] * (self.N + 1)
        
        def update(self, index, value=1):
            while index <= self.N:
                self.tree[index] += value
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res
        
        def query_range(self, left, right):
            return self.query(right) - self.query(left - 1)
    
    # Compute initial inversion count
    bit_initial = BIT(N)
    inv_before = 0
    for i in range(N-1, -1, -1):
        inv_before += bit_initial.query(rank[P[i]] - 1)
        bit_initial.update(rank[P[i]])

    # Precompute adj_inv and prefix_adj_inv
    adj_inv = [0] * (N - 1)
    for j in range(N - 1):
        if P[j] > P[j+1]:
            adj_inv[j] = 1
    prefix_adj_inv = [0] * N
    for j in range(1, N):
        prefix_adj_inv[j] = prefix_adj_inv[j-1] + adj_inv[j-1]

    # Precompute max_prefix
    max_prefix = [0] * (N + 1)
    current_max = 0
    for j in range(1, N+1):
        if P[j-1] > current_max:
            current_max = P[j-1]
        max_prefix[j] = current_max

    # Build BIT for elements after A_i
    bit_after = BIT(N)
    for i in range(1, N+1):
        bit_after.update(rank[P[i-1]])

    removed_ptr = 0
    for a in A:
        # Remove elements from 1 to a from bit_after
        for j in range(removed_ptr + 1, a):
            bit_after.update(rank[P[j-1]], -1)
        removed_ptr = a
        # Get current_max
        current_max = max_prefix[a]
        # Query inv_removed_smaller
        inv_removed_smaller = bit_after.query(rank[current_max] - 1)
        # Query inv_removed_adjacent
        inv_removed_adjacent = prefix_adj_inv[a-1] - prefix_adj_inv[removed_ptr-1] if removed_ptr >=1 else prefix_adj_inv[a-1]
        # Total inv_removed
        inv_removed = inv_removed_smaller + inv_removed_adjacent
        # Update inv_before
        inv_after = inv_before - inv_removed
        print(inv_after)
        inv_before = inv_after

if __name__ == '__main__':
    main()