from typing import List
from array import array

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # ans[l][r] will hold the maximum XOR-score among all subarrays of nums[l..r]
        # Use array('I') for compact storage of unsigned 32-bit ints
        ans = [array('I', [0]) * n for _ in range(n)]
        
        # prev_score[i] = score of subarray nums[i..i+length-2] in the previous iteration
        # Initialize for length = 1
        prev_score = array('I', nums)
        for i in range(n):
            ans[i][i] = prev_score[i]
        
        # Build up for subarray lengths = 2..n
        for length in range(2, n + 1):
            size = n - length + 1
            # curr_score[i] = score of subarray nums[i..i+length-1]
            curr_score = array('I', [0]) * size
            for l in range(size):
                r = l + length - 1
                # recurrence: score[l][r] = score[l][r-1] XOR score[l+1][r]
                curr = prev_score[l] ^ prev_score[l + 1]
                curr_score[l] = curr
                # ans[l][r] = max(ans[l][r-1], ans[l+1][r], curr)
                v1 = ans[l][r - 1]
                v2 = ans[l + 1][r]
                m = v1 if v1 > v2 else v2
                if curr > m:
                    m = curr
                ans[l][r] = m
            prev_score = curr_score
        
        # Answer each query in O(1)
        res = []
        for l, r in queries:
            res.append(ans[l][r])
        return res