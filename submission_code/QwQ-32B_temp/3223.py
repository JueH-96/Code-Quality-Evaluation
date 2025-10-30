class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # Split the word into intervals where all adjacent pairs have difference <=2
        intervals = []
        n = len(word)
        if n == 0:
            return 0
        start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                intervals.append((start, i-1))
                start = i
        intervals.append((start, n-1))  # add the last interval

        total = 0

        for (a, b) in intervals:
            s = word[a : b+1]
            s_len = len(s)
            for m in range(1, 27):  # m from 1 to 26 inclusive
                L = m * k
                if L > s_len:
                    continue
                # Initialize frequency array, count_nonzero, total_valid
                freq = [0] * 26
                count_nonzero = 0
                total_valid = 0
                valid = 0

                # Initialize the first window (0 to L-1)
                for i in range(L):
                    c = s[i]
                    idx = ord(c) - ord('a')
                    prev_count = freq[idx]
                    if prev_count == 0:
                        count_nonzero += 1
                    freq[idx] += 1
                    new_count = prev_count + 1
                    if prev_count == k:
                        total_valid -= 1
                    if new_count == k:
                        total_valid += 1

                if count_nonzero == m and total_valid == m:
                    valid += 1

                # Slide the window
                for i in range(L, s_len):
                    # Remove the left character (i - L)
                    left_char = s[i - L]
                    left_idx = ord(left_char) - ord('a')
                    prev_count_left = freq[left_idx]
                    # Update total_valid and count_nonzero
                    if prev_count_left == k:
                        total_valid -= 1
                    freq[left_idx] -= 1
                    new_count_left = prev_count_left - 1
                    if new_count_left == k:
                        total_valid += 1
                    if new_count_left == 0:
                        count_nonzero -= 1

                    # Add the new right character (i)
                    right_char = s[i]
                    right_idx = ord(right_char) - ord('a')
                    prev_count_right = freq[right_idx]
                    if prev_count_right == k:
                        total_valid -= 1
                    freq[right_idx] += 1
                    new_count_right = prev_count_right + 1
                    if new_count_right == k:
                        total_valid += 1
                    if prev_count_right == 0:
                        count_nonzero += 1

                    # Check validity
                    if count_nonzero == m and total_valid == m:
                        valid += 1

                total += valid

        return total