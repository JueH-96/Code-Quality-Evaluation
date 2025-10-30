import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    
    size = 4 * n
    seg = [0] * size

    def build_tree(idx, l, r):
        if l == r:
            seg[idx] = 1
            return
        mid = (l + r) // 2
        build_tree(2*idx+1, l, mid)
        build_tree(2*idx+2, mid+1, r)
        seg[idx] = seg[2*idx+1] + seg[2*idx+2]
    
    build_tree(0, 0, n-1)
    
    def update_tree_iterative(pos):
        idx = 0
        l = 0
        r = n-1
        while l < r:
            mid = (l + r) // 2
            if pos <= mid:
                idx = 2*idx+1
                r = mid
            else:
                idx = 2*idx+2
                l = mid+1
        seg[idx] = 0
        while idx != 0:
            parent = (idx - 1) // 2
            left_child = 2*parent + 1
            right_child = left_child + 1
            seg[parent] = seg[left_child] + seg[right_child]
            idx = parent
            
    def query_kth(k):
        idx = 0
        l_ptr = 0
        r_ptr = n-1
        while l_ptr < r_ptr:
            mid = (l_ptr + r_ptr) // 2
            left_child_idx = 2*idx + 1
            if seg[left_child_idx] >= k:
                idx = left_child_idx
                r_ptr = mid
            else:
                k -= seg[left_child_idx]
                idx = left_child_idx + 1
                l_ptr = mid + 1
        return l_ptr

    ans = [0] * n
    for num in range(n, 0, -1):
        p_val = P[num-1]
        pos = query_kth(p_val)
        ans[pos] = num
        update_tree_iterative(pos)
        
    print(" ".join(map(str, ans)))

if __name__ == '__main__':
    main()