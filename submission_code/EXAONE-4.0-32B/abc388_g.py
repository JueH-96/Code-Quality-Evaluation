import math
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    k_len = math.floor(math.log2(n)) + 1 if n > 0 else 0
    st = [[0] * n for _ in range(k_len)]
    st[0] = a[:]
    
    for i in range(1, k_len):
        step = 1 << (i-1)
        for j in range(n - (1 << i) + 1):
            st[i][j] = max(st[i-1][j], st[i-1][j+step])
            
    def rmq(l, r):
        if l > r:
            return -10**18
        length = r - l + 1
        t = math.floor(math.log2(length))
        right_start = r - (1 << t) + 1
        return max(st[t][l], st[t][right_start])
    
    def find_first_index(l, r, x):
        low, high = l, r
        ans = None
        while low <= high:
            mid = (low + high) // 2
            if rmq(l, mid) >= x:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

    next_greater = [n] * n
    stack = []
    for i in range(n-1, -1, -1):
        while stack and a[stack[-1]] < 2 * a[i]:
            stack.pop()
        if stack:
            next_greater[i] = stack[-1]
        else:
            next_greater[i] = n
        stack.append(i)
        
    q = int(data[1+n])
    index_ptr = 1 + n + 1
    out_lines = []
    for _ in range(q):
        L = int(data[index_ptr]) - 1
        R = int(data[index_ptr+1]) - 1
        index_ptr += 2
        n_seg = R - L + 1
        k_val = n_seg // 2
        i_ptr = L
        count = 0
        j_ptr = L + k_val
        
        while i_ptr < L + k_val and j_ptr <= R:
            ng = next_greater[i_ptr]
            if ng <= R and ng >= j_ptr:
                jj = ng
            else:
                jj = find_first_index(j_ptr, R, 2 * a[i_ptr])
                if jj is None:
                    break
            count += 1
            i_ptr += 1
            j_ptr = jj + 1
            
        out_lines.append(str(count))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()