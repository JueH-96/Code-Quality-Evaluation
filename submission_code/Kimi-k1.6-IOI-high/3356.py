class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = []
        for i in range(n):
            s = arr[i]
            found = False
            for l in range(1, len(s) + 1):
                substrs = []
                for start in range(len(s) - l + 1):
                    substrs.append(s[start:start+l])
                substrs_sorted = sorted(substrs)
                for sub in substrs_sorted:
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        if sub in arr[j]:
                            valid = False
                            break
                    if valid:
                        answer.append(sub)
                        found = True
                        break
                if found:
                    break
            if not found:
                answer.append("")
        return answer