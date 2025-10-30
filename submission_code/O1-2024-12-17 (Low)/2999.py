class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Helper function to swap characters at positions i and j in a string
        def swap_chars(string, i, j):
            arr = list(string)
            arr[i], arr[j] = arr[j], arr[i]
            return "".join(arr)
        
        # Generate all possible transformations of s1
        transformations = set()
        # 1) The original s1
        transformations.add(s1)
        # 2) After swapping indices (0, 2)
        s_swap_02 = swap_chars(s1, 0, 2)
        transformations.add(s_swap_02)
        # 3) After swapping indices (1, 3)
        s_swap_13 = swap_chars(s1, 1, 3)
        transformations.add(s_swap_13)
        # 4) After swapping indices (0, 2) and (1, 3)
        s_swap_02_13 = swap_chars(s_swap_02, 1, 3)
        transformations.add(s_swap_02_13)
        
        # Check if s2 is one of these transformations
        return s2 in transformations