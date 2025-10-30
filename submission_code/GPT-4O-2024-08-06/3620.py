class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # Sort numbers by frequency (ascending) and then by value (ascending)
        sorted_nums = sorted(freq.items(), key=lambda x: (x[1], x[0]))
        
        # Initialize the number of distinct elements
        distinct_count = 0
        
        # Set to keep track of used numbers
        used_numbers = set()
        
        # Iterate over the sorted numbers
        for num, count in sorted_nums:
            # Try to find a number in the range [num-k, num+k] that is not used
            for delta in range(-k, k + 1):
                new_num = num + delta
                if new_num not in used_numbers:
                    used_numbers.add(new_num)
                    distinct_count += 1
                    break
        
        return distinct_count