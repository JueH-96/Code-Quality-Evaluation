class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # Create a dictionary to store the frequency of each ID
        freq_dict = {}
        for i in range(len(nums)):
            if nums[i] in freq_dict:
                freq_dict[nums[i]] += freq[i]
            else:
                freq_dict[nums[i]] = freq[i]
        
        # Find the maximum frequency
        max_freq = max(freq_dict.values())
        
        # Find the IDs with the maximum frequency
        max_ids = [id for id, freq in freq_dict.items() if freq == max_freq]
        
        # Initialize the result array with zeros
        ans = [0]*len(nums)
        
        # Update the result array
        for i in range(len(nums)):
            if nums[i] in max_ids:
                ans[i] = freq_dict[nums[i]]
        
        return ans