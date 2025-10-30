class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_len = 0
        
        # Create a dictionary to store the frequency of each element
        freq = {}
        
        # Initialize the start and end pointers
        start = 0
        end = 0
        
        # Initialize the count of elements that need to be deleted
        delete_count = 0
        
        while end < n:
            # Increment the frequency of the current element
            freq[nums[end]] = freq.get(nums[end], 0) + 1
            
            # If the current element is not the most frequent element, increment the delete count
            if freq[nums[end]] > (end - start + 1 - k):
                delete_count += 1
            
            # If the delete count exceeds k, move the start pointer forward
            while delete_count > k:
                freq[nums[start]] -= 1
                if freq[nums[start]] <= (end - start + 1 - k):
                    delete_count -= 1
                start += 1
            
            # Update the maximum length of the equal subarray
            max_len = max(max_len, end - start + 1 - delete_count)
            
            # Move the end pointer forward
            end += 1
        
        return max_len