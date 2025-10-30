class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = []
        for i in range(n):
            s = arr[i]
            # Generate all possible substrings for s
            substrings = set()
            for l in range(1, len(s) + 1):
                for start in range(len(s) - l + 1):
                    t = s[start:start+l]
                    substrings.add(t)
            # Check each substring for uniqueness
            candidates = []
            for t in substrings:
                unique = True
                for j in range(n):
                    if i == j:
                        continue
                    if t in arr[j]:
                        unique = False
                        break
                if unique:
                    candidates.append(t)
            if not candidates:
                answer.append('')
                continue
            # Sort the candidates by length and then lexicographically
            candidates.sort(key=lambda x: (len(x), x))
            answer.append(candidates[0])
        return answer