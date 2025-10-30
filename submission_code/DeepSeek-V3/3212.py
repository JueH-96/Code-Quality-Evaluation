class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Create a dictionary to store the last occurrence of each number
        last_occurrence = {}
        for idx, num in enumerate(nums):
            last_occurrence[num] = idx
        
        # Initialize the end of the current partition
        end = 0
        result = 1
        for idx, num in enumerate(nums):
            # Update the end of the current partition
            end = max(end, last_occurrence[num])
            # If the current index is the end of the partition, we can make a cut
            if idx == end:
                result *= 2
                result %= 10**9 + 7
                end += 1
        
        # Since we have one extra cut at the end, we divide by 2
        # But since we are multiplying by 2 each time, the result is already correct
        return result // 2