class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        # Precompute prefix frequency counts for each letter.
        # prefix_count[i] will hold counts for word[0:i]
        prefix_count = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            # Copy previous counts
            prefix_count[i+1] = prefix_count[i].copy()
            prefix_count[i+1][ord(word[i]) - ord('a')] += 1

        # Precompute prefix sum for valid adjacent differences.
        # For index i (1 <= i < n), define diff[i] = 1 if 
        #  abs(word[i] - word[i-1]) <= 2 else 0.
        diff = [0] * n
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) <= 2:
                diff[i] = 1
            else:
                diff[i] = 0
        # Build prefix sum for diff: prefix_diff[i] = sum(diff[1..i-1]) with prefix_diff[0]=0.
        prefix_diff = [0] * (n + 1)
        for i in range(1, n):
            prefix_diff[i + 1] = prefix_diff[i] + diff[i]

        result = 0
        # For each possible number of distinct letters m in the substring.
        # In a complete substring, each character that appears occurs exactly k times,
        # so the substring length L must equal m*k.
        for m in range(1, 27):  # m from 1 to 26
            L = m * k
            if L > n:
                break
            # Slide a window of length L over word.
            for start in range(0, n - L + 1):
                end = start + L - 1  # inclusive end index

                # Check if adjacent difference condition holds:
                # The substring has L-1 adjacent pairs. Its sum in prefix_diff is:
                # prefix_diff[end+1] - prefix_diff[start+1] should equal L-1.
                # (Because diff is stored from indices 1..n-1 in diff[1] corresponds to pair (0,1))
                if prefix_diff[end + 1] - prefix_diff[start + 1] != L - 1:
                    continue

                distinct = 0
                valid = True
                # For each letter, check frequency in the window.
                for c in range(26):
                    count = prefix_count[start + L][c] - prefix_count[start][c]
                    if count != 0 and count != k:
                        valid = False
                        break
                    if count == k:
                        distinct += 1
                if valid and distinct == m:
                    result += 1
        return result