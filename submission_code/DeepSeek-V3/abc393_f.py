import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, Q = map(int, input[ptr:ptr+2])
    ptr +=2
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    queries = []
    all_values = []
    for i in range(Q):
        R = int(input[ptr])
        X = int(input[ptr+1])
        queries.append((R, X, i))
        all_values.append(X)
        ptr +=2
    
    # Coordinate compression
    unique_A = list(set(A))
    all_values.extend(unique_A)
    all_values_sorted = sorted(set(all_values))
    # Create a mapping from value to compressed index
    value_to_compressed = {v: i+1 for i, v in enumerate(all_values_sorted)}
    max_compressed = len(all_values_sorted)
    
    # Process queries offline: sort them by R_i
    queries.sort(key=lambda x: x[0])
    
    # Fenwick Tree for max LIS up to a certain value
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0]*(self.size +2)
        
        def update(self, index, value):
            while index <= self.size:
                if self.tree[index] < value:
                    self.tree[index] = value
                else:
                    break
                index += index & -index
        
        def query(self, index):
            res = 0
            while index >0:
                if self.tree[index] > res:
                    res = self.tree[index]
                index -= index & -index
            return res
    
    fenwick = FenwickTree(max_compressed)
    
    res = [0]*Q
    current_ptr = 0
    for R, X, original_idx in queries:
        while current_ptr < R:
            a = A[current_ptr]
            compressed_a = value_to_compressed[a]
            max_len = fenwick.query(compressed_a -1)
            fenwick.update(compressed_a, max_len +1)
            current_ptr +=1
        # Answer the query: X is compressed
        if X in value_to_compressed:
            compressed_X = value_to_compressed[X]
        else:
            # find the largest value <= X in all_values_sorted
            pos = bisect.bisect_right(all_values_sorted, X) -1
            if pos <0:
                compressed_X =0
            else:
                compressed_X = value_to_compressed[all_values_sorted[pos]]
        ans = fenwick.query(compressed_X)
        res[original_idx] = ans
    
    print('
'.join(map(str, res)))

if __name__ == '__main__':
    main()