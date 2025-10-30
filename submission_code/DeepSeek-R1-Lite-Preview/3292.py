from typing import List
import bisect

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Preprocess: for each index i, collect the list of s where changeIndices[s] = i, 0-indexed
        pos = [[] for _ in range(n)]
        for s in range(m):
            i = changeIndices[s] - 1  # convert to 0-indexed
            pos[i].append(s + 1)  # store 1-indexed seconds
        
        # Binary search over S from 1 to m
        left = 1
        right = m
        answer = -1
        while left <= right:
            mid = (left + right) // 2
            # For each i, find the latest s_i <= mid where changeIndices[s_i] = i
            s_i_list = []
            for i in range(n):
                indices = pos[i]
                # Find the rightmost s <= mid
                idx = bisect.bisect_right(indices, mid) - 1
                if idx < 0:
                    # No s_i found for this i
                    break
                s_i = indices[idx]
                s_i_list.append((s_i, nums[i]))
            else:
                # All indices have s_i
                # Sort the s_i in increasing order
                s_i_list.sort()
                # Now, check if the cumulative nums[i] <= s_i - 1
                cumulative = 0
                feasible = True
                for s_i, num in s_i_list:
                    cumulative += num
                    if cumulative > s_i - 1:
                        feasible = False
                        break
                if feasible:
                    # Try smaller S
                    answer = mid
                    right = mid - 1
                else:
                    # Try larger S
                    left = mid + 1
            if answer != -1:
                # Early exit optimization
                # Try to find a smaller S
                right = mid - 1
            else:
                left = mid + 1
        return answer