class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        # Build requirement counts for word2
        req = [0]*26
        total_deficit = 0
        for ch in word2:
            idx = ord(ch) - ord('a')
            if req[idx] == 0:
                # new requirement
                pass
            req[idx] += 1
            total_deficit += 1

        # current window counts and deficits
        cnt = [0]*26
        deficit = req[:]  # deficit[c] = how many more c's we need
        ans = 0
        r = 0  # window is [l, r), r is exclusive

        for l in range(n):
            # Expand r until we satisfy all deficits or run out
            while r < n and total_deficit > 0:
                c = ord(word1[r]) - ord('a')
                cnt[c] += 1
                # if this character was still needed, reduce deficit
                if deficit[c] > 0:
                    deficit[c] -= 1
                    total_deficit -= 1
                r += 1

            # If total_deficit == 0, window [l, r-1] covers word2 counts.
            # Every substring starting at l and ending at >= r-1 works.
            if total_deficit == 0:
                # r-1 .. n-1 are valid end indices => count = n - (r-1)
                ans += n - r + 1
            else:
                # Can't satisfy any further windows starting at l or beyond
                break

            # Before moving l to l+1, remove word1[l] from window
            c = ord(word1[l]) - ord('a')
            # If removing it causes us to go below requirement, increase deficit
            if cnt[c] > 0:
                if cnt[c] <= req[c]:
                    # we are now short of one
                    deficit[c] += 1
                    total_deficit += 1
                cnt[c] -= 1

        return ans