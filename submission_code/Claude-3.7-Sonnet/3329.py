class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_length = 0
        
        # Convert all numbers to strings once
        arr1_strs = [str(num) for num in arr1]
        arr2_strs = [str(num) for num in arr2]
        
        # Check all pairs
        for str1 in arr1_strs:
            for str2 in arr2_strs:
                # Find the common prefix length for this pair
                prefix_length = 0
                for i in range(min(len(str1), len(str2))):
                    if str1[i] == str2[i]:
                        prefix_length += 1
                    else:
                        break
                
                # Update the maximum length if necessary
                max_length = max(max_length, prefix_length)
        
        return max_length