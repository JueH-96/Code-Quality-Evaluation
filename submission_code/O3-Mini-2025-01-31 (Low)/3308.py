class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Instead of simulating the repeated removals (which would be too slow when rebuilding the string in each round),
        # we can observe the following:
        #
        # In each round, we remove the first occurrence of each letter.
        # Therefore, for each letter, its 1st occurrence is removed in round 1,
        # its 2nd occurrence is removed in round 2, and so on.
        #
        # Let the maximum frequency among all letters be R.
        # Then, the process takes R rounds. In the final (R-th) operation,
        # every occurrence that is the R-th (or later, if available) for that letter is removed.
        # Therefore, the string before the last operation is exactly the collection of characters
        # that are scheduled to be removed in round R (or later).
        #
        # Note that if a letter appears fewer than R times, then all of its occurrences are removed in an earlier round.
        # If R equals 1 (meaning every letter appears only once), then no removals have happened before the final round,
        # and the state before the last operation is just the initial string.
        #
        # We can compute the round (i.e. removal order) for each occurrence by iterating over the string once.
        # Then we include only those characters whose occurrence count for that letter is >= R.
        #
        # For example, for s = "aabcbbca":
        #   a appears 3 times, b appears 3 times, c appears 2 times, so R = 3.
        #   Then, while iterating we label:
        #       a (1st) --> removed in round 1, skip
        #       a (2nd) --> removed in round 2, skip
        #       b (1st) --> removed in round 1, skip
        #       c (1st) --> removed in round 1, skip
        #       b (2nd) --> removed in round 2, skip
        #       b (3rd) --> removed in round 3, so it is the state right before round 3, include "b"
        #       c (2nd) --> removed in round 2, skip (because its removal round < 3)
        #       a (3rd) --> removed in round 3, include "a"
        #   The resulting answer is "ba".
        #
        # This approach processes the string in O(n) time.
        
        # First calculate frequency for each letter to find R (the number of rounds).
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        R = max(freq.values())
        
        # Now, for each letter occurrence, track its occurrence order. The characters that are not removed
        # until the R-th round (i.e. have occurrence count >= R) will be present in the last non-empty state.
        counts = {}
        res = []
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
            if counts[ch] >= R:
                res.append(ch)
        return "".join(res)


# For testing and running the solution by reading from standard input.
def solve():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0]
    sol = Solution()
    sys.stdout.write(sol.lastNonEmptyString(s))
    
if __name__ == '__main__':
    solve()