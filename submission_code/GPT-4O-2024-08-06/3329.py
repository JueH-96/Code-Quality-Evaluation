class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(a: str, b: str) -> int:
            # Find the length of the longest common prefix between two strings
            min_length = min(len(a), len(b))
            for i in range(min_length):
                if a[i] != b[i]:
                    return i
            return min_length
        
        max_length = 0
        
        # Convert all numbers to strings for prefix comparison
        arr1_str = list(map(str, arr1))
        arr2_str = list(map(str, arr2))
        
        # Compare each pair (x, y) where x is from arr1 and y is from arr2
        for x in arr1_str:
            for y in arr2_str:
                max_length = max(max_length, common_prefix_length(x, y))
        
        return max_length