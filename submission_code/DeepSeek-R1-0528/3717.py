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
        while index > 0:
            s += self.tree[index]
            index -= index & -index
        return s

class Solution:
    def minOperations(self, nums: list, x: int, k: int) -> int:
        n_val = len(nums)
        m_val = x // 2
        sorted_vals = sorted(set(nums))
        comp = {val: idx + 1 for idx, val in enumerate(sorted_vals)}
        size_comp = len(sorted_vals)
        
        freq_tree = Fenw(size_comp)
        sum_tree = Fenw(size_comp)
        total_sum = 0
        for i in range(x):
            v = nums[i]
            idx_comp = comp[v]
            freq_tree.update(idx_comp, 1)
            sum_tree.update(idx_comp, v)
            total_sum += v
        
        def query_kth_sum(k_val):
            lo, hi = 1, size_comp
            pos = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                if freq_tree.query(mid) >= k_val:
                    pos = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            freq_so_far = freq_tree.query(pos - 1)
            sum_so_far = sum_tree.query(pos - 1)
            return sum_so_far + sorted_vals[pos - 1] * (k_val - freq_so_far)
        
        n_windows = n_val - x + 1
        costs = [0] * n_windows
        costs[0] = total_sum - query_kth_sum(x - m_val) - query_kth_sum(m_val)
        
        for i in range(1, n_windows):
            v_remove = nums[i - 1]
            idx_remove = comp[v_remove]
            freq_tree.update(idx_remove, -1)
            sum_tree.update(idx_remove, -v_remove)
            total_sum -= v_remove
            
            v_add = nums[i + x - 1]
            idx_add = comp[v_add]
            freq_tree.update(idx_add, 1)
            sum_tree.update(idx_add, v_add)
            total_sum += v_add
            
            costs[i] = total_sum - query_kth_sum(x - m_val) - query_kth_sum(m_val)
        
        INF = 10**18
        dp = [[INF] * (n_val + 1) for _ in range(k + 1)]
        for i in range(n_val + 1):
            dp[0][i] = 0
        
        for j in range(1, k + 1):
            for i in range(n_windows - 1, -1, -1):
                skip = dp[j][i + 1]
                next_index = i + x
                take = costs[i] + dp[j - 1][next_index]
                dp[j][i] = min(skip, take)
        
        return dp[k][0]