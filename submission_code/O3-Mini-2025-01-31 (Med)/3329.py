class Solution:
    def longestCommonPrefix(self, arr1: 'List[int]', arr2: 'List[int]') -> int:
        # Convert the integers in each array to their string representations.
        s1 = [str(num) for num in arr1]
        s2 = [str(num) for num in arr2]
        
        # The maximum length of any qualifying common prefix can only be as long as
        # the shortest number among those that could potentially contribute.
        L_max = min(max(len(s) for s in s1), max(len(s) for s in s2))
        
        # Check for each possible prefix length, starting from the longest possible.
        for L in range(L_max, 0, -1):
            # Only consider numbers that have at least L digits.
            prefixes1 = {s[:L] for s in s1 if len(s) >= L}
            prefixes2 = {s[:L] for s in s2 if len(s) >= L}
            
            # If there is any common prefix of length L, return L immediately.
            if prefixes1 & prefixes2:
                return L
        return 0

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.longestCommonPrefix([1, 10, 100], [1000]))  # expected output: 3
    # Example 2:
    print(sol.longestCommonPrefix([1, 2, 3], [4, 4, 4]))   # expected output: 0