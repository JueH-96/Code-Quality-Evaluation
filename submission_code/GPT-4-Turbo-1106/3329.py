class Solution:
    def longestCommonPrefix(self, arr1: list[int], arr2: list[int]) -> int:
        # Convert all numbers to strings for easier prefix comparison
        arr1 = [str(x) for x in arr1]
        arr2 = [str(x) for x in arr2]
        
        # Sort the arrays to bring similar prefixes together
        arr1.sort()
        arr2.sort()
        
        # Function to find the common prefix length between two strings
        def common_prefix_length(str1, str2):
            i = 0
            while i < min(len(str1), len(str2)) and str1[i] == str2[i]:
                i += 1
            return i
        
        # Initialize the longest common prefix length to 0
        longest = 0
        
        # Compare each element in arr1 with each element in arr2
        for num1 in arr1:
            # Binary search in arr2 for the closest match to maximize the prefix
            left, right = 0, len(arr2) - 1
            while left <= right:
                mid = (left + right) // 2
                # Update the longest common prefix length
                longest = max(longest, common_prefix_length(num1, arr2[mid]))
                # Move the search space
                if arr2[mid] < num1:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return longest