from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        answer = []
        for current_index in range(len(arr)):
            s = arr[current_index]
            substrs = []
            n = len(s)
            # Generate all substrings ordered by length and lex order
            for l in range(1, n + 1):
                current = []
                for i in range(n - l + 1):
                    substr = s[i:i+l]
                    current.append(substr)
                current.sort()
                substrs.extend(current)
            # Check each substring in order
            found = False
            for substr in substrs:
                valid = True
                for j in range(len(arr)):
                    if j == current_index:
                        continue
                    if substr in arr[j]:
                        valid = False
                        break
                if valid:
                    answer.append(substr)
                    found = True
                    break
            if not found:
                answer.append("")
        return answer