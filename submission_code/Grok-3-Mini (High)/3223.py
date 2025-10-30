class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        num_word = [ord(c) - ord('a') for c in word]
        n = len(word)
        ans = 0
        i = 0
        while i < n:
            start = i
            j = i
            while j + 1 < n and abs(num_word[j] - num_word[j + 1]) <= 2:
                j += 1
            # Block from start to j inclusive
            end = j
            block_len = end - start + 1
            max_m = min(26, block_len // k)
            for m in range(1, max_m + 1):
                len_win = m * k
                # Initialize frequency and num_distinct for the first window
                freq = [0] * 26
                num_distinct = 0
                # Add characters from start to start + len_win - 1
                for p in range(start, start + len_win):
                    c = num_word[p]
                    if freq[c] == 0:
                        num_distinct += 1
                    freq[c] += 1
                # Check the first window
                if num_distinct == m and max(freq) <= k:
                    ans += 1
                # Slide the window to the right
                current_left = start
                for new_left in range(start + 1, start + block_len - len_win + 1):
                    # Remove the character at the current left
                    remove_c = num_word[current_left]
                    freq[remove_c] -= 1
                    if freq[remove_c] == 0:
                        num_distinct -= 1
                    # Move left to the new left
                    current_left += 1
                    # Add the character at the new right
                    new_right = current_left + len_win - 1
                    add_c = num_word[new_right]
                    if freq[add_c] == 0:
                        num_distinct += 1
                    freq[add_c] += 1
                    # Check the condition for the new window
                    if num_distinct == m and max(freq) <= k:
                        ans += 1
            # Move to the next block
            i = j + 1
        return ans