import sys
import bisect

class Fenw:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
            
    def query(self, index):
        if index <= 0:
            return 0
        s = 0
        while index:
            s += self.tree[index]
            index -= index & -index
        return s

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:1 + n]))
    B = list(map(int, data[1 + n:1 + 2 * n]))
    k = int(data[1 + 2 * n])
    queries = []
    index_ptr = 1 + 2 * n + 1
    for i in range(k):
        x = int(data[index_ptr])
        y = int(data[index_ptr + 1])
        index_ptr += 2
        queries.append((x, y, i))
        
    all_vals = sorted(set(A) | set(B))
    size_vals = len(all_vals)
    
    def compress(val):
        return bisect.bisect_left(all_vals, val) + 1
        
    cntA_tree = Fenw(size_vals)
    sumA_tree = Fenw(size_vals)
    cntB_tree = Fenw(size_vals)
    sumB_tree = Fenw(size_vals)
    
    sorted_queries = sorted(queries, key=lambda q: (q[1], q[0]))
    
    current_x = 0
    current_y = 0
    total_sum_A = 0
    total_sum_B = 0
    T = 0
    ans_array = [0] * k
    
    for x, y, orig_idx in sorted_queries:
        while current_y < y:
            b_val = B[current_y]
            comp_b = compress(b_val)
            cntA_less = cntA_tree.query(comp_b - 1)
            sumA_less = sumA_tree.query(comp_b - 1)
            cntA_ge = current_x - cntA_less
            sumA_ge = total_sum_A - sumA_less
            T += sumA_ge - b_val * cntA_ge
            cntB_tree.update(comp_b, 1)
            sumB_tree.update(comp_b, b_val)
            total_sum_B += b_val
            current_y += 1
            
        while current_x < x:
            a_val = A[current_x]
            comp_a = compress(a_val)
            cntB_le = cntB_tree.query(comp_a)
            sumB_le = sumB_tree.query(comp_a)
            T += a_val * cntB_le - sumB_le
            cntA_tree.update(comp_a, 1)
            sumA_tree.update(comp_a, a_val)
            total_sum_A += a_val
            current_x += 1
            
        ans = x * total_sum_B - y * total_sum_A + 2 * T
        ans_array[orig_idx] = ans
        
    for ans in ans_array:
        print(ans)

if __name__ == "__main__":
    main()