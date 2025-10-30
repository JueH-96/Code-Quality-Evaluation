class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # The idea is to avoid incurring extra cost by never having the same letter more than once if possible.
        # When we must add a letter that has already appeared (because all 26 letters are used or forced fixed positions duplicate),
        # we choose a letter that, so far, has occurred the least (and in case of a tie, the lexicographically smallest one).
        #
        # To implement this, we process the string from left to right. For each character:
        #   • If it is not '?', we leave it as is and update our record of letters used.
        #   • If it is '?', then:
        #         - If not every letter (a-z) has been used yet, we choose the lexicographically smallest letter that hasn’t been used.
        #         - Otherwise, if all letters have been used,
        #           we choose the letter with the smallest occurrence so far (thus incurring the minimum added cost).
        #
        # This procedure ensures that we incur the smallest possible cost.
        # In addition, because we choose lexicographically smallest options at each decision,
        # the final string is the lexicographically smallest string among all those that achieve the minimum cost.
        
        n = len(s)
        res = []           # list to build the result
        freq = [0] * 26    # frequency counts for letters 'a' to 'z'
        seen = [False] * 26  # to mark which letters have already appeared
        
        # helper function converting char to its index (0 for 'a', 1 for 'b', etc).
        def idx(c):
            return ord(c) - ord('a')
        
        for ch in s:
            if ch != '?':
                # Fixed letter: Append and update frequency and seen.
                res.append(ch)
                j = idx(ch)
                freq[j] += 1
                seen[j] = True
            else:
                # For a '?' we decide which letter to use.
                # Option 1: If not all letters have been seen, we choose the smallest letter not used.
                if not all(seen):
                    for j in range(26):
                        if not seen[j]:
                            c_new = chr(j + ord('a'))
                            res.append(c_new)
                            freq[j] += 1
                            seen[j] = True
                            break
                else:
                    # Option 2: All letters have been used at least once.
                    # Pick the letter with the smallest frequency (if tie, the one that comes first lexicographically).
                    minfreq = float('inf')
                    candidate = None
                    for j in range(26):
                        if freq[j] < minfreq:
                            minfreq = freq[j]
                            candidate = j
                    c_new = chr(candidate + ord('a'))
                    res.append(c_new)
                    freq[candidate] += 1
        
        return "".join(res)


# The following standalone code is provided for testing/command-line usage.
def solve():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    s = data[0].strip()
    sol = Solution()
    result = sol.minimizeStringValue(s)
    sys.stdout.write(result)

if __name__ == '__main__':
    solve()