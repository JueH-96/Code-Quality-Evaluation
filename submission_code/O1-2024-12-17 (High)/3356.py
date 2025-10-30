class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n
        
        # Precompute all strings except the current one for substring checks
        # We can simply check each substring in all other strings using "in" operator
        
        for i in range(n):
            s = arr[i]
            found_substring = ""
            
            # Generate substrings by increasing length
            for length in range(1, len(s) + 1):
                # Collect all substrings of the current length
                subs = []
                for start in range(len(s) - length + 1):
                    subs.append(s[start:start + length])
                
                # Sort lexicographically
                subs.sort()
                
                # Check if a substring is unique to arr[i]
                for sub in subs:
                    is_unique = True
                    for j in range(n):
                        if j == i:
                            continue
                        if sub in arr[j]:
                            is_unique = False
                            break
                    if is_unique:
                        found_substring = sub
                        break
                
                if found_substring:
                    break
            
            answer[i] = found_substring
        
        return answer