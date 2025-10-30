class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        initial_or = 0
        for num in nums:
            initial_or |= num
        
        low = 0
        high = initial_or
        answer = initial_or
        
        while low <= high:
            mid = (low + high) // 2
            
            non_subset = 0
            has_subset = False
            for num in nums:
                if (num | mid) != mid:
                    non_subset += 1
                else:
                    has_subset = True
            
            if non_subset == 0:
                # All elements are subsets; possible.
                answer = mid
                high = mid - 1
            else:
                if has_subset:
                    if non_subset <= k:
                        answer = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                else:
                    low = mid + 1
        
        return answer