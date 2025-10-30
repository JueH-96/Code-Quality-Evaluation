import bisect

class SegmentTree:
    def __init__(self, size):
        if size == 0:
            self.tree = None
            self.n = 0
            self.size = 0
        else:
            self.n = 1
            while self.n < size:
                self.n <<= 1
            self.size = size
            self.tree = [-float('inf')] * (2 * self.n)
    
    def update_point(self, idx, value):
        if self.tree is None:
            return
        idx += self.n
        if self.tree[idx] >= value:
            return
        self.tree[idx] = value
        while idx > 1:
            idx >>= 1
            new_val = max(self.tree[2 * idx], self.tree[2 * idx + 1])
            if self.tree[idx] == new_val:
                break
            self.tree[idx] = new_val
    
    def query_range(self, l, r):
        if self.tree is None:
            return -float('inf')
        res = -float('inf')
        l += self.n
        r += self.n
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    nums1 = list(map(int, input[ptr:ptr + n]))
    ptr += n
    nums2 = list(map(int, input[ptr:ptr + n]))
    ptr += n
    queries = []
    for i in range(int(input[ptr])):
        x = int(input[ptr + 1])
        y = int(input[ptr + 2])
        ptr += 3
        queries.append((x, y))
    
    j_list = []
    for j in range(n):
        a_j = nums1[j]
        b_j = nums2[j]
        s_j = a_j + b_j
        j_list.append((a_j, b_j, s_j))
    
    j_list.sort(key=lambda x: (-x[0], -x[2]))
    
    b_values = [j[1] for j in j_list]
    sorted_b = sorted(b_values)
    if not sorted_b:
        st = None
    else:
        st = SegmentTree(len(sorted_b))
    
    queries_sorted = sorted(queries, key=lambda x: (-x[0], -x[1]))
    result = []
    ptr_j = 0
    
    for x, y in queries_sorted:
        while ptr_j < len(j_list) and j_list[ptr_j][0] >= x:
            a_j, b_j, s_j = j_list[ptr_j]
            r = bisect.bisect_left(sorted_b, b_j)
            if r < len(sorted_b):
                if st is not None:
                    st.update_point(r, s_j)
            ptr_j += 1
        
        if st is None:
            result.append(-1)
        else:
            r = bisect.bisect_left(sorted_b, y)
            if r >= len(sorted_b):
                result.append(-1)
            else:
                max_s = st.query_range(r, len(sorted_b) - 1)
                if max_s == -float('inf'):
                    result.append(-1)
                else:
                    result.append(max_s)
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()