class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        result = []
        
        for i in range(n):
            current_string = arr[i]
            found = False
            
            # Try substrings of increasing length
            for length in range(1, len(current_string) + 1):
                candidates = []
                
                for start in range(len(current_string) - length + 1):
                    substring = current_string[start:start + length]
                    
                    # Check if this substring occurs in any other string
                    found_in_other = False
                    for j in range(n):
                        if i != j and substring in arr[j]:
                            found_in_other = True
                            break
                    
                    if not found_in_other:
                        candidates.append(substring)
                
                if candidates:
                    # Found valid substrings of this length
                    result.append(min(candidates))
                    found = True
                    break
            
            if not found:
                result.append("")
        
        return result