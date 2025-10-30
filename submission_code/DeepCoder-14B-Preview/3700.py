from itertools import combinations
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def subsequencesWithMiddleMode(self, nums: list) -> int:
        n = len(nums)
        result = 0
        
        for i in range(n):
            x = nums[i]
            pre = nums[:i]
            suf = nums[i+1:]
            
            pre_count = defaultdict(int)
            for num in pre:
                pre_count[num] += 1
            
            suf_count = defaultdict(int)
            for num in suf:
                suf_count[num] += 1
            
            # Calculate the number of ways to choose two elements before i
            pre_len = len(pre)
            ways_pre = 0
            for t in range(3):
                if t > pre.count(x):
                    continue
                for comb in combinations(pre, 2):
                    count_x = sum(1 for num in comb if num == x)
                    current_x = 1 + count_x
                    max_other = 0
                    others = [num for num in comb if num != x]
                    for y in others:
                        cnt = others.count(y)
                        if cnt > max_other:
                            max_other = cnt
                    if current_x > max_other:
                        ways_pre += 1
            
            # Calculate the number of ways to choose two elements after i
            suf_len = len(suf)
            ways_suf = 0
            for s in range(3):
                if s > suf.count(x):
                    continue
                for comb in combinations(suf, 2):
                    count_x = sum(1 for num in comb if num == x)
                    current_x = 1 + count_x
                    max_other = 0
                    others = [num for num in comb if num != x]
                    for y in others:
                        cnt = others.count(y)
                        if cnt > max_other:
                            max_other = cnt
                    if current_x > max_other:
                        ways_suf += 1
            
            result += ways_pre * ways_suf
            result %= MOD
        
        return result