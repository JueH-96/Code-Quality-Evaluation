class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def all_substrings(s):
            n = len(s)
            return [s[i:j] for i in range(n) for j in range(i + 1, n + 1)]
        
        n = len(arr)
        answer = [''] * n
        
        # Collect all substrings of each string
        substrings = [all_substrings(s) for s in arr]
        
        # Check each substring for each string
        for i in range(n):
            current_string = arr[i]
            candidate = None
            
            # Sort substrings by length and lexicographically
            sorted_substrings = sorted(set(substrings[i]), key=lambda x: (len(x), x))
            
            for sub in sorted_substrings:
                found_in_other = False
                for j in range(n):
                    if i != j and sub in arr[j]:
                        found_in_other = True
                        break
                
                if not found_in_other:
                    candidate = sub
                    break
            
            if candidate:
                answer[i] = candidate
            else:
                answer[i] = ''
        
        return answer