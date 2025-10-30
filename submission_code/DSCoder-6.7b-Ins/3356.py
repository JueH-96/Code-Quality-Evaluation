class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def find_substrings(s):
            substrings = set()
            for i in range(len(s)):
                for j in range(i + 1, len(s) + 1):
                    substrings.add(s[i:j])
            return substrings

        answer = []
        for s in arr:
            substrings = find_substrings(s)
            for sub in substrings:
                if not any(sub in ss for ss in arr if ss != s):
                    answer.append(sub)
                    break
            else:
                answer.append("")
        return answer