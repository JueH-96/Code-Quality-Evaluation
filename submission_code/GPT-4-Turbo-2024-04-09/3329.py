class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all numbers to strings for easier prefix comparison
        arr1 = [str(x) for x in arr1]
        arr2 = [str(x) for x in arr2]
        
        # Sort arrays to utilize the property that common prefixes will be adjacent in sorted order
        arr1.sort()
        arr2.sort()
        
        # Function to find the longest common prefix between two strings
        def common_prefix_length(s1, s2):
            min_len = min(len(s1), len(s2))
            for i in range(min_len):
                if s1[i] != s2[i]:
                    return i
            return min_len
        
        # Initialize the maximum length of common prefix
        max_length = 0
        
        # Compare each element in arr1 with each element in arr2
        # Since arrays are sorted, we can use binary search to find the closest elements
        import bisect
        for num1 in arr1:
            # Binary search to find the closest element in arr2
            idx = bisect.bisect_left(arr2, num1)
            
            # Check with the found index and the previous index (if exists)
            if idx < len(arr2):
                max_length = max(max_length, common_prefix_length(num1, arr2[idx]))
            if idx > 0:
                max_length = max(max_length, common_prefix_length(num1, arr2[idx - 1]))
        
        return max_length