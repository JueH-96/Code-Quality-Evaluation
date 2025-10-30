from typing import List

class Solution:
    MOD = 10**9 + 7

    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        self.size = 1
        while self.size < n:
            self.size <<= 1
        self.tree = [self.create_leaf(0) for _ in range(2 * self.size)]
        for i in range(n):
            self.tree[self.size + i] = self.create_leaf(nums[i])
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.merge(self.tree[2 * i], self.tree[2 * i + 1])
        res = 0
        for pos, x in queries:
            i = pos
            self.tree[self.size + i] = self.create_leaf(x)
            i += self.size
            i >>= 1
            while i >= 1:
                self.tree[i] = self.merge(self.tree[2 * i], self.tree[2 * i + 1])
                i >>= 1
            current_max = max(self.tree[1]['max_sum'], 0)
            res = (res + current_max) % self.MOD
        return res % self.MOD

    def create_leaf(self, val):
        return {
            't_f_t_l': val,
            't_f_n_l': -float('inf'),
            'n_t_f_t_l': -float('inf'),
            'n_t_f_n_l': 0,
            'max_sum': max(val, 0)
        }

    def merge(self, left, right):
        merged = {}
        merged['t_f_t_l'] = -float('inf')
        a = left['t_f_t_l'] + right['n_t_f_n_l'] if left['t_f_t_l'] != -float('inf') else -float('inf')
        b = left['t_f_n_l'] + max(right['t_f_n_l'], right['n_t_f_n_l']) if left['t_f_n_l'] != -float('inf') else -float('inf')
        merged['t_f_n_l'] = max(a, b)
        c = right['t_f_t_l'] + left['n_t_f_n_l'] if right['t_f_t_l'] != -float('inf') else -float('inf')
        d = right['n_t_f_t_l'] + max(left['t_f_n_l'], left['n_t_f_n_l']) if right['n_t_f_t_l'] != -float('inf') else -float('inf')
        merged['n_t_f_t_l'] = max(c, d)
        merged['n_t_f_n_l'] = max(left['n_t_f_n_l'], left['n_t_f_t_l']) + max(right['t_f_n_l'], right['n_t_f_n_l'])
        merged['max_sum'] = max(merged['t_f_t_l'], merged['t_f_n_l'], merged['n_t_f_t_l'], merged['n_t_f_n_l'])
        return merged