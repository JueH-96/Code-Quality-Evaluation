mod = 998244353

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] = (self.tree[index] + delta) % mod
            index += index & -index
            
    def query(self, index):
        s = 0
        while index > 0:
            s = (s + self.tree[index]) % mod
            index -= index & -index
        return s
    
    def range_query(self, l, r):
        if l > r:
            return 0
        res = self.query(r) - self.query(l - 1)
        return res % mod

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    P = list(map(int, data[2:2+n]))
    
    T = n - k + 1
    invT = pow(T, mod - 2, mod)
    invK = pow(k, mod - 2, mod)
    inv4 = pow(4, mod - 2, mod)
    
    inv0 = 0
    fenw_global = Fenw(n)
    for i in range(n - 1, -1, -1):
        inv0 = (inv0 + fenw_global.query(P[i])) % mod
        fenw_global.update(P[i], 1)
    
    if k == 1:
        print(0)
        return
        
    total_within = 0
    window_inv = 0
    fenw_window = Fenw(n)
    for i in range(k):
        window_inv = (window_inv + fenw_window.query(n) - fenw_window.query(P[i])) % mod
        fenw_window.update(P[i], 1)
    total_within = (total_within + window_inv) % mod
    
    for l in range(1, T):
        r = l + k - 1
        remove_val = P[l - 1]
        loss = fenw_window.query(remove_val - 1)
        fenw_window.update(remove_val, -1)
        window_inv = (window_inv - loss) % mod
        
        add_val = P[r]
        gain = fenw_window.query(n) - fenw_window.query(add_val)
        fenw_window.update(add_val, 1)
        window_inv = (window_inv + gain) % mod
        total_within = (total_within + window_inv) % mod

    left_part_arr = [0] * T
    fenw_left = Fenw(n)
    fenw_current_vals = Fenw(n)
    current_left_part = 0

    for val in P[:k]:
        fenw_current_vals.update(val, 1)
    for l in range(T):
        left_part_arr[l] = current_left_part
        if l < n - 1:
            remove_val = P[l]
            count_greater_left = fenw_left.query(n) - fenw_left.query(remove_val)
            current_left_part = (current_left_part - count_greater_left) % mod
            fenw_current_vals.update(remove_val, -1)
            
            add_to_left = P[l]
            count_less_in_current = fenw_current_vals.query(add_to_left - 1)
            current_left_part = (current_left_part + count_less_in_current) % mod
            fenw_left.update(add_to_left, 1)
            
            if l + k < n:
                add_to_seg = P[l + k]
                count_greater_left_for_new = fenw_left.query(n) - fenw_left.query(add_to_seg)
                current_left_part = (current_left_part + count_greater_left_for_new) % mod
                fenw_current_vals.update(add_to_seg, 1)
                
    right_part_arr = [0] * T
    fenw_right = Fenw(n)
    fenw_current_vals_right = Fenw(n)
    current_right_part = 0
    for val in P[-k:]:
        fenw_current_vals_right.update(val, 1)
    for l in range(T - 1, -1, -1):
        right_part_arr[l] = current_right_part
        if l > 0:
            remove_val = P[l + k - 1]
            count_less_right = fenw_right.query(remove_val - 1)
            current_right_part = (current_right_part - count_less_right) % mod
            fenw_current_vals_right.update(remove_val, -1)
            
            add_to_right = P[l + k - 1]
            count_greater_in_current = fenw_current_vals_right.query(n) - fenw_current_vals_right.query(add_to_right)
            current_right_part = (current_right_part + count_greater_in_current) % mod
            fenw_right.update(add_to_right, 1)
            
            if l - 1 >= 0:
                add_to_seg = P[l - 1]
                count_less_right_for_new = fenw_right.query(add_to_seg - 1)
                current_right_part = (current_right_part + count_less_right_for_new) % mod
                fenw_current_vals_right.update(add_to_seg, 1)
                
    total_cross = 0
    for i in range(T):
        total_cross = (total_cross + left_part_arr[i] + right_part_arr[i]) % mod

    term2 = T * k % mod * (k - 1) % mod * inv4 % mod
    term3 = (k - 1) * invK % mod * total_cross % mod
    term3 = (-term3) % mod

    numerator = (inv0 * T % mod - total_within - total_cross + term2 + term3) % mod
    ans = numerator * invT % mod
    print(ans % mod)

if __name__ == '__main__':
    main()