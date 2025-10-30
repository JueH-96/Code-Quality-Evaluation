class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        valid_starts = []
        
        # Check each possible starting index for a valid increasing subarray
        for a in range(n - k + 1):
            valid = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid = False
                    break
            if valid:
                valid_starts.append(a)
        
        # Convert to a set for O(1) lookups
        valid_set = set(valid_starts)
        
        # Check if any start index has another valid start exactly k positions after it
        for a in valid_starts:
            if (a + k) in valid_set:
                return True
        
        return False