from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        result = [""] * n
        string_to_index = defaultdict(set)
        
        # Populate the dictionary with string occurrences
        for i, s in enumerate(arr):
            for j in range(len(s)):
                for k in range(j + 1, len(s) + 1):
                    string_to_index[s[j:k]].add(i)
        
        # Find the shortest unique substring for each string
        for i, s in enumerate(arr):
            for j in range(len(s)):
                for k in range(j + 1, len(s) + 1):
                    if len(string_to_index[s[j:k]]) == 1:
                        result[i] = s[j:k]
                        break
                if result[i]:
                    break
        
        return result