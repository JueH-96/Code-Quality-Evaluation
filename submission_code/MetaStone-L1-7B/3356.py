class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        result = []
        for s in arr:
            candidates = []
            for k in range(1, len(s) + 1):
                sub_k = [s[i:i+k] for i in range(len(s) - k + 1)]
                for t in sub_k:
                    valid = True
                    for other in arr:
                        if other == s:
                            continue
                        if t in other:
                            valid = False
                            break
                    if valid:
                        candidates.append(t)
                if candidates:
                    result.append(min(candidates))
                    break
            else:
                result.append("")
        return result