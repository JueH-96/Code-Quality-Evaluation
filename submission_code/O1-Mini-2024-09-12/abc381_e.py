# YOUR CODE HERE
import sys
import threading
from bisect import bisect_left, bisect_right

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict

    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    queries = []
    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())
        queries.append((L, R))
    
    prefix1 = [0] * (N + 1)
    prefix2 = [0] * (N +1)
    for i in range(1, N+1):
        prefix1[i] = prefix1[i-1]
        prefix2[i] = prefix2[i-1]
        if S[i-1] == '1':
            prefix1[i] +=1
        elif S[i-1] == '2':
            prefix2[i] +=1

    # Collect all slash positions
    slashes = []
    for pos in range(1, N+1):
        if S[pos-1] == '/':
            slashes.append(pos)
    
    # Precompute prefix1[pos-1] and prefix2[pos] for slashes
    slash_info = []
    for pos in slashes:
        slash_info.append( (prefix1[pos-1], prefix2[pos], pos) )
    # Sort slashes descendingly by prefix1[pos-1]
    slash_info.sort(reverse=True, key=lambda x: (x[0], -x[2]))
    
    # Binary Indexed Tree (Fenwick Tree) for range min queries
    class BIT:
        def __init__(self, size):
            self.N = size
            self.tree = [math.inf] * (self.N +2)
        
        def update(self, idx, val):
            while idx <= self.N:
                if val < self.tree[idx]:
                    self.tree[idx] = val
                else:
                    break
                idx += idx & -idx
        
        def query_min(self, l, r):
            return self._query(r) - self._query(l-1)
        
        def _query(self, idx):
            res = math.inf
            while idx >0:
                res = min(res, self.tree[idx])
                idx -= idx & -idx
            return res

    # Segment Tree for range min queries
    class SegTree:
        def __init__(self, size):
            self.N = 1
            while self.N < size:
                self.N <<=1
            self.size = self.N
            self.inf = math.inf
            self.data = [self.inf] * (2*self.N)
        
        def update(self, pos, val):
            pos += self.N -1
            if val < self.data[pos]:
                self.data[pos] = val
                pos >>=1
                while pos >=1:
                    self.data[pos] = min(self.data[pos<<1], self.data[(pos<<1)|1])
                    pos >>=1
            # else do nothing
        
        def query_min(self, l, r):
            res = self.inf
            l += self.N -1
            r += self.N -1
            while l <= r:
                if l%2 ==1:
                    res = min(res, self.data[l])
                    l +=1
                if r%2 ==0:
                    res = min(res, self.data[r])
                    r -=1
                l >>=1
                r >>=1
            return res

    # Initialize binary search parameters
    low = [0] * Q
    high = [0] * Q
    for i in range(Q):
        L, R = queries[i]
        total1 = prefix1[R] - prefix1[L-1]
        total2 = prefix2[R] - prefix2[L-1]
        high[i] = min(total1, total2)
    
    answer = [0] * Q
    active = True
    while active:
        active = False
        mid_to_queries = defaultdict(list)
        for i in range(Q):
            if low[i] < high[i]:
                active = True
                mid = (low[i] + high[i] +1) //2
                mid_to_queries[mid].append(i)
        if not active:
            break
        sorted_mid = sorted(mid_to_queries.keys(), reverse=True)
        # Sort slashes descendingly by prefix1[pos-1]
        # Already sorted in slash_info
        # We will iterate slashes in order and insert into segment tree as needed
        st = SegTree(N)
        ptr =0
        for k in sorted_mid:
            # For all queries with mid=k, we need to have A = prefix1[L-1] +k and B=prefix2[R] -k
            # Insert into segment tree all slashes with prefix1[pos-1] >= A for these queries
            # But different A for different queries
            # So, process all queries with mid=k together
            # To handle multiple A, process queries sorted by A descendingly
            queries_k = mid_to_queries[k]
            # Sort queries_k by A descendingly
            queries_k_sorted = sorted(queries_k, key=lambda x: prefix1[queries[x][0]-1] +k, reverse=True)
            for q in queries_k_sorted:
                L, R = queries[q]
                A = prefix1[L-1] +k
                B = prefix2[R] -k
                # Insert into segment tree all slashes with prefix1[pos-1] >=A
                while ptr < len(slash_info) and slash_info[ptr][0] >= A:
                    pos = slash_info[ptr][2]
                    st.update(pos, slash_info[ptr][1])
                    ptr +=1
                # Query the segment tree in [L,R] for min prefix2[pos] <= B
                min_val = st.query_min(L, R)
                if min_val <= B:
                    low[q] = k
                else:
                    high[q] = k-1
        # End of this binary search step
    for i in range(Q):
        if low[i] >0:
            answer[i] = 2*low[i] +1
        elif low[i]==0:
            # Check if a '/' itself is valid for length=1
            L, R = queries[i]
            # If there is a '/' in [L,R], len=1 is possible
            # But only if k=0, which always holds, so answer=1
            # Unless no '/' exists in [L,R]
            # However, per the definition, k=0 means only the '/' is present, which is acceptable
            # So, if there's at least one '/' in [L,R], answer=1
            # Else, 0
            # So, set answer=1 if there exists '/' in [L,R], else 0
            # We can check this using binary search on slashes sorted
            # slashes are sorted in any order, but we can sort them ascendingly for binary search
            # So, pre-sort slashes ascendingly
            pass
    # To handle cases where low[i]==0, need to check if there's at least one '/' in [L,R]
    # Sort slashes ascendingly
    sorted_slashes_asc = sorted(slashes)
    for i in range(Q):
        if low[i] ==0:
            L, R = queries[i]
            # Binary search to find if any slash in [L,R]
            left = bisect_left(sorted_slashes_asc, L)
            right = bisect_right(sorted_slashes_asc, R)
            if left < right:
                answer[i] =1
            else:
                answer[i] =0
    for a in answer:
        print(a)

threading.Thread(target=main).start()