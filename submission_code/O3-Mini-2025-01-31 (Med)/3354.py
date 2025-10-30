class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # We want to replace every '?' in s with a letter so that the total cost is minimal.
        # The cost for index i is the number of times t[i] has already appeared in t[0...i-1].
        # Notice that if we can avoid repeating a letter (i.e. make it the first occurrence),
        # its cost contribution is 0. So to minimize cost we want each letter at its first occurrence.
        #
        # However, fixed letters in s force some costs. When we get a '?' we have a choice:
        #   - If there is any letter that has not yet appeared (frequency 0), replacing '?' with such a letter
        #     gives cost 0 at that position.
        #   - If all letters have already appeared, any replacement will add the current frequency of that letter
        #     (which is at least 1). In that case we want to choose the letter with the smallest current frequency,
        #     and among those, the lexicographically smallest choice.
        #
        # This greedy strategy works because the cost added at any position is independent given the prefix counts,
        # and our modifications only affect future choices through frequency updates.
        
        res = []  # List to build the final string.
        freq = {chr(ord('a') + i): 0 for i in range(26)}
        
        for ch in s:
            if ch != '?':
                res.append(ch)
                freq[ch] += 1
            else:
                # First try to find a letter that has not yet appeared.
                candidate = None
                for letter in "abcdefghijklmnopqrstuvwxyz":
                    if freq[letter] == 0:
                        candidate = letter
                        break
                # If every letter has appeared before, then choose the one with the smallest frequency.
                if candidate is None:
                    minFreq = float('inf')
                    for letter in "abcdefghijklmnopqrstuvwxyz":
                        if freq[letter] < minFreq:
                            minFreq = freq[letter]
                            candidate = letter
                res.append(candidate)
                freq[candidate] += 1
        
        return "".join(res)

# Optional: Code for running the solution with standard input.
if __name__ == '__main__':
    import sys
    data = sys.stdin.read().splitlines()
    if data:
        s = data[0].strip()
        solution = Solution()
        result = solution.minimizeStringValue(s)
        sys.stdout.write(result)