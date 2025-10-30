import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # Using 1-based indexing

    def update(self, idx, delta):
        idx += 1  # Convert to 1-based index
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        idx += 1  # Convert to 1-based index
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
    ptr +=1
    K = int(input[ptr])
    ptr +=1
    Q = int(input[ptr])
    ptr +=1

    queries = []
    all_values = set()
    for _ in range(Q):
        X = int(input[ptr])
        ptr +=1
        Y = int(input[ptr])
        ptr +=1
        queries.append( (X,Y) )
        all_values.add(Y)

    # Add initial zeros
    all_values.add(0)
    # Convert to a sorted list
    U = sorted(all_values)
    value_to_idx = {v:i for i,v in enumerate(U)}

    # Initialize Fenwick Trees
    size_ft = len(U)
    count_tree = FenwickTree(size_ft)
    sum_tree = FenwickTree(size_ft)

    # Initial setup: all elements are zero
    initial_val = 0
    initial_idx = value_to_idx[initial_val]
    count_tree.update(initial_idx, N)
    sum_tree.update(initial_idx, initial_val * N)

    # Initialize array
    A = [0] * N

    for X, Y in queries:
        x = X - 1  # Convert to 0-based index
        old_val = A[x]
        new_val = Y

        old_idx = value_to_idx[old_val]
        new_idx = value_to_idx[new_val]

        # Update Fenwick Trees
        count_tree.update(old_idx, -1)
        sum_tree.update(old_idx, -old_val)

        count_tree.update(new_idx, 1)
        sum_tree.update(new_idx, new_val)

        A[x] = new_val

        # Binary search to find V
        low = 0
        high = len(U) -1
        while low < high:
            mid = (low + high +1) // 2
            cnt = count_tree.query(len(U)-1) - count_tree.query(mid-1)
            if cnt >= K:
                low = mid
            else:
                high = mid -1
        V_idx = low
        V = U[V_idx]

        # Calculate total sum and count
        total_sum = sum_tree.query(len(U)-1) - sum_tree.query(V_idx-1)
        total_count = count_tree.query(len(U)-1) - count_tree.query(V_idx-1)

        if total_count <= K:
            print(total_sum)
        else:
            res = total_sum - (total_count - K) * V
            print(res)

if __name__ == '__main__':
    main()