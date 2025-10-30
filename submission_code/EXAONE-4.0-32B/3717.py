class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
            
    def query(self, index):
        s = 0
        while index:
            s += self.tree[index]
            index -= index & -index
        return s

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        n_segments = n - x + 1
        if k == 0:
            return 0
        
        sorted_vals = sorted(set(nums))
        comp_map = {}
        for idx, val in enumerate(sorted_vals):
            comp_map[val] = idx + 1
        d = len(sorted_vals)
        
        fenw_freq = Fenw(d)
        fenw_sum = Fenw(d)
        
        for i in range(x):
            num = nums[i]
            cidx = comp_map[num]
            fenw_freq.update(cidx, 1)
            fenw_sum.update(cidx, num)
            
        costs = [0] * n_segments
        
        def get_cost():
            kth = (x + 1) // 2
            low, high = 1, d
            while low < high:
                mid = (low + high) // 2
                if fenw_freq.query(mid) < kth:
                    low = mid + 1
                else:
                    high = mid
            pos = low
            median_val = sorted_vals[pos - 1]
            count_L = fenw_freq.query(pos)
            sum_L = fenw_sum.query(pos)
            total_sum = fenw_sum.query(d)
            count_R = x - count_L
            sum_R = total_sum - sum_L
            return (median_val * count_L - sum_L) + (sum_R - median_val * count_R)
        
        costs[0] = get_cost()
        for i in rangen_segments - 1):
            cidx_remove = comp_map[nums[i]]
            fenw_freq.update(cidx_remove, -1)
            fenw_sum.update(cidx_remove, -nums[i])
            
            cidx_add = comp_map[nums[i + x]]
            fenw_freq.update(cidx_add, 1)
            fenw_sum.update(cidx_add, nums[i + x])
            
            costs[i + 1] = get_cost()
        
        INF = 10**18
        dp_prev = [0] * n_segments
        min_arr_prev = [0] * n_segments
        
        for j in range(1, k + 1):
            dp_curr = [INF] * n_segments
            min_arr_curr = [INF] * n_segments
            for i in range(n_segments):
                if j == 1:
                    candidate = costs[i]
                else:
                    if i < x:
                        candidate = INF
                    else:
                        candidate = costs[i] + min_arr_prev[i - x]
                if i >= 1:
                    skip_val = dp_curr[i - 1]
                else:
                    skip_val = INF
                dp_curr[i] = min(skip_val, candidate)
                if i == 0:
                    min_arr_curr[i] = dp_curr[i]
                else:
                    min_arr_curr[i] = min(min_arr_curr[i - 1], dp_curr[i])
            dp_prev = dp_curr
            min_arr_prev = min_arr_curr
        
        return min_arr_prev[-1]