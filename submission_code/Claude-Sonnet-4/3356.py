class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_all_substrings(s):
            substrings = []
            for i in range(len(s)):
                for j in range(i + 1, len(s) + 1):
                    substrings.append(s[i:j])
            return substrings
        
        def is_substring_in_other_strings(substring, target_idx, arr):
            for i, s in enumerate(arr):
                if i != target_idx and substring in s:
                    return True
            return False
        
        result = []
        
        for i, s in enumerate(arr):
            substrings = get_all_substrings(s)
            # Sort by length first, then lexicographically
            substrings.sort(key=lambda x: (len(x), x))
            
            found = False
            for substring in substrings:
                if not is_substring_in_other_strings(substring, i, arr):
                    result.append(substring)
                    found = True
                    break
            
            if not found:
                result.append("")
        
        return result