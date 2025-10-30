class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        best = None  # will hold the lexicographically smallest candidate of minimal length
        min_length = float('inf')
        
        # iterate over all substrings
        for i in range(n):
            ones_count = 0
            # expand substring ending at j from i to n-1
            for j in range(i, n):
                if s[j] == '1':
                    ones_count += 1
                # once we hit exactly k ones, check candidate
                if ones_count == k:
                    curr_length = j - i + 1
                    candidate = s[i:j+1]
                    if curr_length < min_length:
                        min_length = curr_length
                        best = candidate
                    elif curr_length == min_length:
                        # update if lexicographically smaller
                        if candidate < best:
                            best = candidate
                    # We don't break because further extension might yield the same count
                    # (if additional characters are '0') and might be lexicographically smaller.
        return best if best is not None else ""