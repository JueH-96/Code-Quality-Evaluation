class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        root = {}
        for num in arr1:
            s = str(num)
            node = root
            for char in s:
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        max_len = 0
        for num in arr2:
            s = str(num)
            node = root
            cur_len = 0
            for char in s:
                if char in node:
                    cur_len += 1
                    node = node[char]
                else:
                    break
            if cur_len > max_len:
                max_len = cur_len
        return max_len