class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        
        # Convert nums to a pattern array that represents relationships between adjacent elements
        nums_pattern = []
        for i in range(n - 1):
            if nums[i+1] > nums[i]:
                nums_pattern.append(1)
            elif nums[i+1] == nums[i]:
                nums_pattern.append(0)
            else:
                nums_pattern.append(-1)
        
        # Compute the LPS (Longest Prefix Suffix) array for KMP algorithm
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        # Use KMP algorithm to find pattern matches
        count = 0
        i = 0  # index for nums_pattern
        j = 0  # index for pattern
        
        while i < len(nums_pattern):
            if nums_pattern[i] == pattern[j]:
                i += 1
                j += 1
                
                if j == m:
                    count += 1
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return count