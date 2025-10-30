class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        best_str = None

        for m in range(1, numFriends + 1):
            window_start = m - 1
            window_end = n - numFriends + m - 1
            L = window_end - window_start + 1

            if L <= 0:
                continue

            # Compute the starting index of the maximum suffix in the current window
            i = 0
            j = 1
            k = 0
            while j + k < L:
                a = word[window_start + i + k]
                b = word[window_start + j + k]
                if a == b:
                    k += 1
                elif a < b:
                    i = j
                    j += 1
                    k = 0
                else:
                    j += k + 1
                    k = 0
                # Ensure j does not fall behind i
                if i + k >= j:
                    j += 1

            best_start = window_start + i
            current_substring = word[best_start:window_end + 1]

            if best_str is None or current_substring > best_str:
                best_str = current_substring

        return best_str