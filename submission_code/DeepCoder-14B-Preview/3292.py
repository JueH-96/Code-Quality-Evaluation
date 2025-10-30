class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Check if all indices from 1 to n are present in changeIndices
        present = set(changeIndices)
        for j in range(1, n + 1):
            if j not in present:
                return -1
        
        # Binary search on the possible values of s
        low = 1
        high = m
        answer = -1
        
        while low <= high:
            mid = (low + high) // 2
            if self.is_possible(nums, changeIndices, mid, n):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer if answer != -1 else -1
    
    def is_possible(self, nums, changeIndices, s, n):
        current_nums = list(nums)
        marked = [False] * n
        
        for i in range(s):
            j = changeIndices[i] - 1  # convert to 0-based index
            
            if not marked[j] and current_nums[j] == 0:
                marked[j] = True
            else:
                # Try to decrement the current j if possible
                if not marked[j] and current_nums[j] > 0:
                    current_nums[j] -= 1
                else:
                    # Find any other unmarked index to decrement
                    found = False
                    for k in range(n):
                        if not marked[k] and current_nums[k] > 0:
                            current_nums[k] -= 1
                            found = True
                            break
                    if not found:
                        pass
        
        return all(marked)