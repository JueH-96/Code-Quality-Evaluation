class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary to store the frequency of possible ranges
        freq = defaultdict(int)
        
        # For each number, calculate the range it can be modified to
        for num in nums:
            start = num - k
            end = num + k
            # Increment the frequency of each range
            freq[(start, end)] += 1
        
        # The maximum beauty is the maximum frequency of any range
        max_beauty = max(freq.values())
        
        return max_beauty