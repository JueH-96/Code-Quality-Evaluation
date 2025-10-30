from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        sum_nums = sum(nums)
        left = 1
        right = m
        answer = -1
        
        while left <= right:
            mid = (left + right) // 2
            last_occ = [0] * (n + 1)  # 1-based indexing for 1..n
            
            # Populate last_occ with the last occurrence of each index up to mid
            for s in range(mid):
                current_idx = changeIndices[s]
                last_occ[current_idx] = s + 1  # Convert 0-based s to 1-based second
            
            # Check if all indices have at least one occurrence
            all_have = True
            for i in range(1, n + 1):
                if last_occ[i] == 0:
                    all_have = False
                    break
            
            if not all_have:
                left = mid + 1
                continue
            
            # Check if each nums[i] can be reduced to 0 before its last_occ[i]
            valid = True
            for i in range(1, n + 1):
                if nums[i - 1] > (last_occ[i] - 1):
                    valid = False
                    break
            
            if not valid:
                left = mid + 1
                continue
            
            # Check if total operations (sum_nums + n) can fit into mid seconds
            total_ops = sum_nums + n
            if total_ops <= mid:
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer if answer != -1 else -1