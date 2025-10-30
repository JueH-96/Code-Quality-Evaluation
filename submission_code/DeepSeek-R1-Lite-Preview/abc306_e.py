import sys
import bisect

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K, Q = map(int, sys.stdin.readline().split())
    updates = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    
    # Collect all unique Y values
    Y_values = set()
    for x, y in updates:
        Y_values.add(y)
    # Sort Y in descending order and assign ranks
    sorted_Y = sorted(Y_values, reverse=True)
    rank_dict = {y: i+1 for i, y in enumerate(sorted_Y)}
    size = len(sorted_Y) + 2  # for 1-based indexing

    # Segment tree implementation
    class SegmentTree:
        def __init__(self, size):
            self.N = 1
            while self.N < size:
                self.N <<= 1
            self.size = self.N
            self.tree = [0] * (2 * self.N)
        
        def update_point(self, idx, value):
            idx += self.N - 1  # 1-based indexing
            self.tree[idx] = value
            while idx > 1:
                idx >>= 1
                self.tree[idx] = self.tree[2*idx] + self.tree[2*idx+1]
        
        def query_range(self, l, r):
            res = 0
            l += self.N - 1
            r += self.N - 1
            while l <= r:
                if l % 2 == 1:
                    res += self.tree[l]
                    l += 1
                if r % 2 == 0:
                    res += self.tree[r]
                    r -= 1
                l >>= 1
                r >>= 1
            return res
    
    st = SegmentTree(size)
    A = [0] * (N + 1)  # 1-based indexing

    # Since initially all A[i] are 0, and 0 may not be in sorted_Y
    # We need to handle 0 appropriately
    if 0 in rank_dict:
        zero_rank = rank_dict[0]
        st.update_point(zero_rank, N)
    else:
        sorted_Y.append(0)
        sorted_Y.sort(reverse=True)
        rank_dict = {y: i+1 for i, y in enumerate(sorted_Y)}
        size = len(sorted_Y) + 2
        st = SegmentTree(size)
        zero_rank = rank_dict[0]
        st.update_point(zero_rank, N)
    
    for x, y in updates:
        old_val = A[x]
        new_val = y
        A[x] = new_val
        # Get rank of new_val
        if new_val in rank_dict:
            new_rank = rank_dict[new_val]
        else:
            # Insert new_val into sorted_Y and update rank_dict
            bisect.insort_left(sorted_Y, new_val, lo=0, hi=len(sorted_Y), key=lambda a: -a)
            rank = bisect.bisect_left(sorted_Y, new_val, lo=0, hi=len(sorted_Y), key=lambda a: -a) + 1
            rank_dict[new_val] = rank
            # Update segment tree size if necessary
            if rank > st.size:
                # Increase the size of the segment tree
                while st.size < rank:
                    st.size <<= 1
                # Reinitialize the tree
                st = SegmentTree(st.size)
                # Rebuild the tree with existing frequencies
                freq = {}
                for val in sorted_Y:
                    r = rank_dict[val]
                    freq[r] = freq.get(r, 0) + A.count(val)
                for r in range(1, len(sorted_Y)+1):
                    st.update_point(r, sorted_Y[r-1] * freq.get(r, 0))
            new_rank = rank_dict[new_val]
        
        # Get rank of old_val
        if old_val != 0:
            old_rank = rank_dict[old_val]
            # Decrease frequency of old_val
            st.update_point(old_rank, sorted_Y[old_rank-1] * (st.query_range(old_rank, old_rank) // sorted_Y[old_rank-1] - 1))
        else:
            old_rank = zero_rank
            st.update_point(old_rank, sorted_Y[old_rank-1] * (st.query_range(old_rank, old_rank) // sorted_Y[old_rank-1] - 1))
        
        # Increase frequency of new_val
        st.update_point(new_rank, sorted_Y[new_rank-1] * (st.query_range(new_rank, new_rank) // sorted_Y[new_rank-1] + 1))
        
        # Query sum of top K elements
        if K <= 0:
            print(0)
        else:
            sum_top_K = st.query_range(1, min(K, len(sorted_Y)))
            print(sum_top_K)

if __name__ == '__main__':
    main()