class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_freq = 0
        
        # For each possible starting index
        for i in range(n):
            # For each possible ending index
            for j in range(i, n):
                # Calculate what x needs to be added to make nums[i] become k
                x = k - nums[i]
                
                # Count how many elements would become k after adding x
                count = 0
                for idx in range(n):
                    # If index is within range [i,j], add x to the element
                    if i <= idx <= j:
                        if nums[idx] + x == k:
                            count += 1
                    # If index is outside range [i,j], check if element is already k
                    else:
                        if nums[idx] == k:
                            count += 1
                
                max_freq = max(max_freq, count)
        
        return max_freq