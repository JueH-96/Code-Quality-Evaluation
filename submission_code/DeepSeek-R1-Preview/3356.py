class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        answer = []
        n = len(arr)
        for i in range(n):
            s = arr[i]
            other_strings = arr[:i] + arr[i+1:]
            other_subs = set()
            # Generate all substrings from other strings
            for other in other_strings:
                m = len(other)
                for l in range(1, m + 1):
                    for j in range(m - l + 1):
                        substr = other[j:j+l]
                        other_subs.add(substr)
            # Generate candidates for the current string in the required order
            candidates = []
            m = len(s)
            for l in range(1, m + 1):
                current_substrings = []
                for j in range(m - l + 1):
                    current_substrings.append(s[j:j+l])
                current_substrings.sort()
                candidates.extend(current_substrings)
            # Find the first candidate not in other_subs
            res = ""
            for sub in candidates:
                if sub not in other_subs:
                    res = sub
                    break
            answer.append(res)
        return answer