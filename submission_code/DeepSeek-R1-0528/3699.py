import collections
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        left_freq = collections.defaultdict(int)
        left_freq[nums[0]] = 1
        res = 0
        
        for q in range(2, n-4):
            if q > 2:
                left_freq[nums[q-2]] += 1
            
            right_freq = collections.defaultdict(int)
            for s in range(q+4, n):
                right_freq[nums[s]] += 1
            
            num_q = nums[q]
            for r in range(q+2, n-2):
                num_r = nums[r]
                if len(left_freq) < len(right_freq):
                    for a, cnt_a in left_freq.items():
                        product = a * num_r
                        if product % num_q == 0:
                            b = product // num_q
                            cnt_b = right_freq.get(b, 0)
                            if cnt_b:
                                res += cnt_a * cnt_b
                else:
                    for b, cnt_b in right_freq.items():
                        product = num_q * b
                        if product % num_r == 0:
                            a = product // num_r
                            cnt_a = left_freq.get(a, 0)
                            if cnt_a:
                                res += cnt_a * cnt_b
                
                s_val = nums[r+2]
                right_freq[s_val] -= 1
                if right_freq[s_val] == 0:
                    del right_freq[s_val]
        
        return res