class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # First, we need to identify the dominant element in nums.
        # We know there is exactly one dominant element by problem statement.
        
        # STEP 1: Find candidate for majority (dominant) element using Boyer-Moore voting.
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        
        # STEP 2: Confirm the candidate is indeed the dominant element by counting its total frequency.
        total_freq = nums.count(candidate)
        
        # STEP 3: We need freq(candidate)*2 > len(nums) which is guaranteed by the statement,
        # but we'll rely on it for splitting. We'll track how many times 'candidate' appears
        # up to each index i.
        
        left_count = 0  # frequency of candidate in the left part so far
        n = len(nums)
        
        for i in range(n - 1):  # valid i: 0 <= i < n-1
            if nums[i] == candidate:
                left_count += 1
            
            left_size = i + 1
            right_size = n - (i + 1)
            
            # Check if 'candidate' is dominant in the left subarray and right subarray.
            # left subarray: [0..i], size = left_size
            # freq_in_left = left_count
            # Condition: freq_in_left * 2 > left_size
            
            # right subarray: [i+1..n-1], size = right_size
            # freq_in_right = total_freq - left_count
            # Condition: freq_in_right * 2 > right_size
            
            if left_count * 2 > left_size and (total_freq - left_count) * 2 > right_size:
                return i
        
        return -1