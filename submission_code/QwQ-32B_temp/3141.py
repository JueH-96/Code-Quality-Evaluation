from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        S = sum(nums)
        L = len(nums)
        if S == 0:
            return -1  # As per constraints, this won't happen
        
        # Precompute the doubled array and its prefix sums and prefix_map
        arr = nums + nums
        prefix = [0]
        prefix_map = {0: -1}  # key: prefix_sum, value: earliest index (in arr) where this sum occurs
        
        for i in range(len(arr)):
            current_sum = prefix[-1] + arr[i]
            prefix.append(current_sum)
            if current_sum not in prefix_map:
                prefix_map[current_sum] = i
        
        def find_min(X):
            min_len = float('inf')
            for j in range(len(arr)):
                current_sum = prefix[j + 1]
                needed = current_sum - X
                if needed in prefix_map:
                    i = prefix_map[needed]
                    length = j - i
                    if length < min_len:
                        min_len = length
            return min_len if min_len != float('inf') else -1
        
        if target < S:
            min_len = find_min(target)
            return min_len if min_len != -1 else -1
        else:
            m = target // S
            rem = target % S
            if rem == 0:
                return m * L
            else:
                min1 = find_min(rem)
                X2 = rem + S
                min2 = find_min(X2)
                
                candidate1 = (m * L) + min1 if min1 != -1 else float('inf')
                candidate2 = ((m - 1) * L) + min2 if min2 != -1 else float('inf')
                
                res = min(candidate1, candidate2)
                return res if res != float('inf') else -1