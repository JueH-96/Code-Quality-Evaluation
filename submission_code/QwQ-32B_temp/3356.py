from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        all_substrings = []  # Each element is a set of substrings for arr[i]
        
        # Precompute all substrings for each string
        for s in arr:
            subs = set()
            m = len(s)
            for l in range(1, m + 1):
                for start in range(m - l + 1):
                    substr = s[start:start+l]
                    subs.add(substr)
            all_substrings.append(subs)
        
        answer = []
        for i in range(n):
            forbidden = set()
            # Collect all substrings from other strings
            for j in range(n):
                if j != i:
                    forbidden.update(all_substrings[j])
            
            current = arr[i]
            m = len(current)
            current_subs = set()
            # Generate all substrings of current string
            for l in range(1, m + 1):
                for start in range(m - l + 1):
                    sub = current[start:start+l]
                    current_subs.add(sub)
            
            # Sort substrings by length then lex order
            sorted_subs = sorted(current_subs, key=lambda x: (len(x), x))
            
            res = ""
            for sub in sorted_subs:
                if sub not in forbidden:
                    res = sub
                    break
            answer.append(res)
        
        return answer