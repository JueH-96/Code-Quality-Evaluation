class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        
        # Calculate prefix sums for efficient range sum queries
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        def getCost(left, right):
            # Cost to make all elements in [left, right] equal to the median
            length = right - left + 1
            mid = (left + right) // 2
            
            # For odd length, median is nums[mid]
            # For even length, any value between nums[mid] and nums[mid+1] works
            # We use nums[mid] for simplicity
            
            # Cost = sum of distances to median
            # Elements before median need to increase
            # Elements after median need to decrease
            
            # Sum of elements from left to mid-1
            leftSum = prefix[mid] - prefix[left]
            # Count of elements from left to mid-1
            leftCount = mid - left
            # Cost to increase them to nums[mid]
            leftCost = nums[mid] * leftCount - leftSum
            
            # Sum of elements from mid+1 to right
            rightSum = prefix[right + 1] - prefix[mid + 1]
            # Count of elements from mid+1 to right
            rightCount = right - mid
            # Cost to decrease them to nums[mid]
            rightCost = rightSum - nums[mid] * rightCount
            
            return leftCost + rightCost
        
        maxFreq = 1
        left = 0
        
        for right in range(n):
            # Shrink window while cost exceeds k
            while left < right and getCost(left, right) > k:
                left += 1
            
            maxFreq = max(maxFreq, right - left + 1)
        
        return maxFreq