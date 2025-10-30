from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        max_num = 10**6
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:  # i is a prime
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        n = len(nums)
        if n == 0:
            return 0
        
        possible = []
        for x in nums:
            if x == 1:
                possible.append([1])
                continue
            values = []
            current = x
            while True:
                values.append(current)
                if current == spf[current]:
                    break
                current = spf[current]
            possible.append(values)
        
        # Initialize DP for the first element
        prev_dp = list(range(len(possible[0])))
        
        for i in range(1, n):
            curr_possible = possible[i]
            prev_possible = possible[i-1]
            m_prev = len(prev_possible)
            
            # Compute suffix min for prev_dp
            suffix_min_prev = [0] * m_prev
            suffix_min_prev[-1] = prev_dp[-1]
            for j in range(m_prev - 2, -1, -1):
                suffix_min_prev[j] = min(prev_dp[j], suffix_min_prev[j + 1])
            
            len_curr = len(curr_possible)
            curr_dp = [float('inf')] * len_curr
            
            for idx in range(len_curr):
                v = curr_possible[idx]
                low, high = 0, m_prev - 1
                res = -1
                # Binary search in prev_possible for first index where value <= v
                while low <= high:
                    mid = (low + high) // 2
                    if prev_possible[mid] <= v:
                        res = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                if res != -1:
                    curr_dp[idx] = suffix_min_prev[res] + idx
            
            min_val = min(curr_dp)
            if min_val == float('inf'):
                return -1
            prev_dp = curr_dp
        
        return min(prev_dp)