class Solution:
    def shortestSubstrings(self, arr: list[str]) -> list[str]:
        def is_unique_substring(s, index):
            for i, other in enumerate(arr):
                if i != index and s in other:
                    return False
            return True

        answer = []
        for i, string in enumerate(arr):
            found = False
            for length in range(1, len(string) + 1):
                substrings = sorted(set(string[j:j+length] for j in range(len(string) - length + 1)))
                for substring in substrings:
                    if is_unique_substring(substring, i):
                        answer.append(substring)
                        found = True
                        break
                if found:
                    break
            if not found:
                answer.append("")
        return answer