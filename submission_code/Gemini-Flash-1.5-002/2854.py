class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        def join_strings(s1, s2):
            if s1 and s2 and s1[-1] == s2[0]:
                return s1[:-1] + s2
            return s1 + s2

        def solve(idx, current_string):
            if idx == len(words):
                return len(current_string)

            option1 = solve(idx + 1, join_strings(current_string, words[idx]))
            option2 = solve(idx + 1, join_strings(words[idx], current_string))
            return min(option1, option2)

        return solve(1, words[0])