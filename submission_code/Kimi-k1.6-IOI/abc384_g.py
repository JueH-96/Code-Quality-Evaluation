import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.count_tree = [0] * (self.size + 2)  # 1-based indexing
        self.sum_tree = [0] * (self.size + 2)
    
    def update(self, idx, val):
        i = idx
        while i <= self.size:
            self.count_tree[i] += 1
            self.sum_tree[i] += val
            i += i & -i
    
    def query(self, idx):
        count = 0
        sum_val = 0
        i = idx
        while i > 0:
            count += self.count_tree[i]
            sum_val += self.sum_tree[i]
            i -= i & -i
        return (count, sum_val)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    K = int(input[ptr])
    ptr += 1
    queries = []
    for i in range(K):
        X = int(input[ptr])
        Y = int(input[ptr+1])
        queries.append((X, Y, i))
        ptr += 2
    
    # Coordinate compression
    all_values = A + B
    sorted_unique = sorted(list(set(all_values)))
    rank = {val: i+1 for i, val in enumerate(sorted_unique)}
    compressed_A = [rank[val] for val in A]
    compressed_B = [rank[val] for val in B]
    
    # Prefix sums
    prefix_A = [0] * (N + 1)
    for i in range(N):
        prefix_A[i+1] = prefix_A[i] + A[i]
    prefix_B = [0] * (N + 1)
    for i in range(N):
        prefix_B[i+1] = prefix_B[i] + B[i]
    
    # Split queries
    group1 = []  # X <= Y
    group2 = []  # X > Y
    for q in queries:
        X, Y, idx = q
        if X <= Y:
            group1.append((Y, X, idx))
        else:
            group2.append((X, Y, idx))
    
    group1.sort()
    group2.sort()
    
    max_rank = len(sorted_unique)
    results = [0] * K
    
    # Process group1 (X <= Y)
    fenwick_B = FenwickTree(max_rank)
    ptr1 = 0
    for Y in range(1, N+1):
        original_val = B[Y-1]
        compressed_val = compressed_B[Y-1]
        fenwick_B.update(compressed_val, original_val)
        while ptr1 < len(group1) and group1[ptr1][0] == Y:
            Yq, Xq, idx = group1[ptr1]
            sum_min = 0
            for i in range(Xq):
                a_val = A[i]
                a_rank = compressed_A[i]
                cnt, s = fenwick_B.query(a_rank)
                sum_min += s + a_val * (Yq - cnt)
            total = Xq * prefix_B[Yq] + Yq * prefix_A[Xq] - 2 * sum_min
            results[idx] = total
            ptr1 += 1
    
    # Process group2 (X > Y)
    fenwick_A = FenwickTree(max_rank)
    ptr2 = 0
    for X in range(1, N+1):
        original_val = A[X-1]
        compressed_val = compressed_A[X-1]
        fenwick_A.update(compressed_val, original_val)
        while ptr2 < len(group2) and group2[ptr2][0] == X:
            Xq, Yq, idx = group2[ptr2]
            sum_min = 0
            for j in range(Yq):
                b_val = B[j]
                b_rank = compressed_B[j]
                cnt, s = fenwick_A.query(b_rank)
                sum_min += s + b_val * (Xq - cnt)
            total = Xq * prefix_B[Yq] + Yq * prefix_A[Xq] - 2 * sum_min
            results[idx] = total
            ptr2 += 1
    
    for res in results:
        print(res)

if __name__ == '__main__':
    main()