from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 1:
            return len(words[0])

        dp = {}

        def solve(idx, current_str):
            if idx == n:
                return len(current_str)
            
            if (idx, current_str[-1] if current_str else "") in dp:
                return dp[(idx, current_str[-1] if current_str else "")]

            ans = float('inf')
            
            # Option 1: join(current_str, words[idx])
            if current_str:
                if current_str[-1] == words[idx][0]:
                    ans = min(ans, solve(idx + 1, current_str + words[idx][1:]))
                else:
                    ans = min(ans, solve(idx + 1, current_str + words[idx]))
            else:
                ans = min(ans, solve(idx+1, words[idx]))
            
            # Option 2: join(words[idx], current_str)
            if current_str:
                if words[idx][-1] == current_str[0]:
                    ans = min(ans, solve(idx + 1, words[idx] + current_str[1:]))
                else:
                    ans = min(ans, solve(idx + 1, words[idx] + current_str))
            else:
                ans = min(ans, solve(idx+1, words[idx]))
            
            dp[(idx, current_str[-1] if current_str else "")] = ans
            return ans

        return solve(1, words[0])