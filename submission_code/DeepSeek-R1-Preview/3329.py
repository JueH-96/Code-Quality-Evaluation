class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Build a trie from the numbers in arr1
        trie = {}
        for num in arr1:
            s = str(num)
            current = trie
            for c in s:
                if c not in current:
                    current[c] = {}
                current = current[c]
        
        max_len = 0
        
        # Check each number in arr2 against the trie to find the maximum prefix length
        for num in arr2:
            s = str(num)
            current = trie
            current_len = 0
            for c in s:
                if c in current:
                    current_len += 1
                    current = current[c]
                else:
                    break
            if current_len > max_len:
                max_len = current_len
        
        return max_len