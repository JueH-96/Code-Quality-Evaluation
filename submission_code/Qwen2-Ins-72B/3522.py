from collections import deque

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        dq = deque()
        
        for i in range(n):
            # Remove elements that are out of the current window
            while dq and dq[0] < i - k + 1:
                dq.popleft()
            
            # Remove elements that are smaller than the current element from the back of the deque
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            # Add the current index to the deque
            dq.append(i)
            
            # If the window size is reached, check if the elements are consecutive and sorted
            if i >= k - 1:
                if nums[dq[0]] + k - 1 == nums[dq[-1]]:
                    results.append(nums[dq[-1]])
                else:
                    results.append(-1)
        
        return results