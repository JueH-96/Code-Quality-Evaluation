import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
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
    
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    k = int(next(it))
    queries = []
    for i in range(k):
        x = int(next(it))
        y = int(next(it))
        queries.append((x, y, i))
        
    sorted_A = sorted(A)
    sorted_B = sorted(B)
    
    all_vals = sorted(set(A) | set(B))
    comp_map = {}
    for idx, val in enumerate(all_vals):
        comp_map[val] = idx + 1
        
    m = len(all_vals)
    
    queries_sorted = sorted(queries, key=lambda q: (q[0], q[1]))
    
    fenw_A_count = Fenw(m)
    fenw_A_sum = Fenw(m)
    fenw_B_count = Fenw(m)
    fenw_B_sum = Fenw(m)
    
    current_x = 0
    current_y = 0
    global_T = 0
    global_U = 0
    total_count_A = 0
    total_sum_A = 0
    total_count_B = 0
    total_sum_B = 0
    ans_arr = [0] * k
    
    for (x, y, idx) in queries_sorted:
        while current_x < x:
            a_val = sorted_A[current_x]
            pos = comp_map[a_val]
            count_b = fenw_B_count.query(pos)
            sum_b = fenw_B_sum.query(pos)
            global_T += a_val * count_b
            global_U += sum_b
            fenw_A_count.update(pos, 1)
            fenw_A_sum.update(pos, a_val)
            total_count_A += 1
            total_sum_A += a_val
            current_x += 1
            
        while current_x > x:
            a_val = sorted_A[current_x - 1]
            pos = comp_map[a_val]
            count_b = fenw_B_count.query(pos)
            sum_b = fenw_B_sum.query(pos)
            global_T -= a_val * count_b
            global_U -= sum_b
            fenw_A_count.update(pos, -1)
            fenw_A_sum.update(pos, -a_val)
            total_count_A -= 1
            total_sum_A -= a_val
            current_x -= 1
            
        while current_y < y:
            b_val = sorted_B[current_y]
            pos = comp_map[b_val]
            if pos - 1 >= 1:
                count_ge = total_count_A - fenw_A_count.query(pos - 1)
                sum_ge = total_sum_A - fenw_A_sum.query(pos - 1)
            else:
                count_ge = total_count_A
                sum_ge = total_sum_A
            global_T += sum_ge
            global_U += b_val * count_ge
            fenw_B_count.update(pos, 1)
            fenw_B_sum.update(pos, b_val)
            total_count_B += 1
            total_sum_B += b_val
            current_y += 1
            
        while current_y > y:
            b_val = sorted_B[current_y - 1]
            pos = comp_map[b_val]
            if pos - 1 >= 1:
                count_ge = total_count_A - fenw_A_count.query(pos - 1)
                sum_ge = total_sum_A - fenw_A_sum.query(pos - 1)
            else:
                count_ge = total_count_A
                sum_ge = total_sum_A
            global_T -= sum_ge
            global_U -= b_val * count_ge
            fenw_B_count.update(pos, -1)
            fenw_B_sum.update(pos, -b_val)
            total_count_B -= 1
            total_sum_B -= b_val
            current_y -= 1
            
        SA = total_sum_A
        SB = total_sum_B
        res = 2 * global_T - y * SA + x * SB - 2 * global_U
        ans_arr[idx] = res
        
    for i in range(k):
        print(ans_arr[i])
        
if __name__ == "__main__":
    main()