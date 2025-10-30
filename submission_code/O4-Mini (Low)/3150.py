class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = None
        
        for i in range(n):
            count_ones = 0
            # Expand the window from i to j
            for j in range(i, n):
                if s[j] == '1':
                    count_ones += 1
                # Once we exceed k ones, no need to go further from this i
                if count_ones > k:
                    break
                # When we hit exactly k ones, consider this substring
                if count_ones == k:
                    curr_len = j - i + 1
                    substr = s[i:j+1]
                    if curr_len < min_len:
                        min_len = curr_len
                        result = substr
                    elif curr_len == min_len and substr < result:
                        result = substr
                    # We can break here if we only care about the first time we reach k ones,
                    # because further j will only make the substring longer while count_ones stays == k or grows.
                    break
        
        return result if result is not None else ""