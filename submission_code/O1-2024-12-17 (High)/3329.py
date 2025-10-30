class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Build a dictionary of all prefixes from arr1
        # The key will be the integer prefix itself, and 
        # the value will be the length (number of digits) of that prefix.
        prefix_map = {}
        
        for num in arr1:
            s = str(num)
            p = 0
            for i, ch in enumerate(s):
                # Build the prefix integer without repeated conversions
                p = p * 10 + (ord(ch) - ord('0'))
                # Store the length if it's the first time we see this prefix
                if p not in prefix_map:
                    prefix_map[p] = i + 1
        
        # Now check prefixes from arr2 against prefix_map
        result = 0
        for num in arr2:
            s = str(num)
            p = 0
            for i, ch in enumerate(s):
                p = p * 10 + (ord(ch) - ord('0'))
                # If this prefix exists in arr1's map, update result
                if p in prefix_map:
                    result = max(result, prefix_map[p])
                else:
                    # If this prefix does not exist in arr1, 
                    # no longer prefixes for this number can match
                    break
        
        return result