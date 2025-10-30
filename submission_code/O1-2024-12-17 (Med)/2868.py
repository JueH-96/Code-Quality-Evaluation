class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        from collections import deque
        
        minQ = deque()  # will store candidates for the minimum in ascending order
        maxQ = deque()  # will store candidates for the maximum in descending order
        
        left = 0
        result = 0
        
        for right in range(len(nums)):
            # Maintain maxQ in descending order
            while maxQ and maxQ[-1] < nums[right]:
                maxQ.pop()
            maxQ.append(nums[right])
            
            # Maintain minQ in ascending order
            while minQ and minQ[-1] > nums[right]:
                minQ.pop()
            minQ.append(nums[right])
            
            # Shrink window if difference exceeds 2
            while maxQ[0] - minQ[0] > 2:
                if maxQ[0] == nums[left]:
                    maxQ.popleft()
                if minQ[0] == nums[left]:
                    minQ.popleft()
                left += 1
            
            # Number of valid subarrays ending at right
            result += (right - left + 1)
        
        return result