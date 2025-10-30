class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        result = [""] * n
        
        for i in range(n):
            s = arr[i]
            for length in range(1, len(s) + 1):
                min_substr = None
                
                for start in range(len(s) - length + 1):
                    substr = s[start:start + length]
                    is_unique = True
                    
                    for j in range(n):
                        if i != j and substr in arr[j]:
                            is_unique = False
                            break
                    
                    if is_unique and (min_substr is None or substr < min_substr):
                        min_substr = substr
                
                if min_substr:
                    result[i] = min_substr
                    break
        
        return result