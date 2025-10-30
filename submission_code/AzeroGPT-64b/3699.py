from collections import Counter, defaultdict
from sortedcontainers import SortedList

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        """
        Find the number of special subsequences in the given list of numbers.
        A special subsequence of length 4 (p, q, r, s) must satisfy:
        nums[p] * nums[r] == nums[q] * nums[s]
        q - p > 1, r - q > 1, and s - r > 1
        """
        INVALID = -1
        product_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        product_search = defaultdict(lambda: defaultdict(lambda: defaultdict(INVALID)))
        idx_by_num = defaultdict(SortedList)
        num_sum = defaultdict(int)
        
        for i, num in enumerate(nums):
            idx_by_num[num].add(i)
            num_sum[num] += 1
        
        for r, num in enumerate(nums):          
            for a in range(1, num + 1):
                for b in range(1, num + 1):
                    if num * a * b <= 1000000:
                        c = (num * a * b) // (a * b)
                        if c <= 1000:
                            first_a = idx_by_num[a].bisect_left(r + 2)
                            product_search[r][a][b] = idx_by_num[c].bisect_left(first_a + 2)
        for i, num in enumerate(nums):
            for a in range(1, num + 1):
                for b in range(1, num + 1):
                    if num * a * b <= 1000000:
                        c = (num * a * b) // (a * b)
                        if c <= 1000:
                            product_sum[i][a][b] = (num_sum[c] - 
                                                product_search[i - 1][a][b] if i > 0 else 0 -
                                                product_search[i][a][b])
        
        answer = 0
        for i in range(len(nums)): 
            num = nums[i] 
            if num != 1: 
                answer += product_sum[i][num][num]
            for a in range(1, num): 
                b = (a * num) // (num - a)
                if num * a * b <= 1000000: 
                    c = (num * a * b) // (a * b)
                    if c <= 1000:
                        answer += product_sum[i][a][b]
        
        return answer // 2