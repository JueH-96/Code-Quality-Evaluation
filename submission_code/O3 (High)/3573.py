class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # frequency of every character required by word2
        need = [0] * 26
        for ch in word2:
            need[ord(ch) - 97] += 1

        n = len(word1)
        # quick exit – impossible to satisfy
        if len(word2) > n:
            return 0

        # current window frequencies
        have = [0] * 26
        # how many characters are still missing to satisfy `need`
        missing = len(word2)

        left = 0           # left border of the sliding window
        ans = 0            # result

        for right in range(n):
            r_idx = ord(word1[right]) - 97
            have[r_idx] += 1
            if need[r_idx] > 0 and have[r_idx] <= need[r_idx]:
                missing -= 1        # one required character satisfied

            # when the window already covers all requirements
            if missing == 0:
                # make the window minimal by throwing away
                # unnecessary characters from the left side
                while left <= right:
                    l_idx = ord(word1[left]) - 97
                    # we can remove word1[left] if we still have enough
                    if have[l_idx] > need[l_idx]:
                        have[l_idx] -= 1
                        left += 1
                    else:
                        break

                # left is the minimal starting index of a covering window.
                # Any start position in [0 .. left] together with current `right`
                # forms a valid substring, hence (left + 1) new substrings.
                ans += left + 1

        return ans