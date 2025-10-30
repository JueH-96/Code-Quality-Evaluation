from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        answer = []
        for i in range(len(arr)):
            exclude = set()
            # Collect all substrings from other strings
            for j in range(len(arr)):
                if i == j:
                    continue
                s = arr[j]
                n = len(s)
                for l in range(1, n + 1):
                    for start in range(n - l + 1):
                        substr = s[start:start + l]
                        exclude.add(substr)
            # Process current string arr[i]
            current = arr[i]
            min_sub = ""
            found = False
            for l in range(1, len(current) + 1):
                subs = []
                n_current = len(current)
                for start in range(n_current - l + 1):
                    subs.append(current[start:start + l])
                # Sort lexicographically
                subs_sorted = sorted(subs)
                for sub in subs_sorted:
                    if sub not in exclude:
                        min_sub = sub
                        found = True
                        break
                if found:
                    break
            answer.append(min_sub if found else "")
        return answer