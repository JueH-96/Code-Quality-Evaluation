from bisect import bisect_left
from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all integers to strings
        s1 = [str(x) for x in arr1]
        s2 = [str(x) for x in arr2]
        
        # Sort both lists
        s1.sort()
        s2.sort()
        
        # We'll choose the smaller list to iterate over, and binary search in the larger list
        if len(s1) > len(s2):
            s_small, s_large = s2, s1
        else:
            s_small, s_large = s1, s2
        
        def common_prefix_length(a: str, b: str) -> int:
            cp = 0
            ln = min(len(a), len(b))
            for i in range(ln):
                if a[i] == b[i]:
                    cp += 1
                else:
                    break
            return cp

        max_common = 0
        
        for s in s_small:
            # Find the insertion point in the large list sorted order
            pos = bisect_left(s_large, s)
            # Check the element at pos if it exists.
            if pos < len(s_large):
                cp = common_prefix_length(s, s_large[pos])
                if cp > max_common:
                    max_common = cp
            # Also check the previous element in the sorted list if it exists.
            if pos > 0:
                cp = common_prefix_length(s, s_large[pos-1])
                if cp > max_common:
                    max_common = cp
        
        return max_common

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.longestCommonPrefix([1,10,100], [1000]))  # Expected output: 3
    print(sol.longestCommonPrefix([1,2,3], [4,4,4]))      # Expected output: 0