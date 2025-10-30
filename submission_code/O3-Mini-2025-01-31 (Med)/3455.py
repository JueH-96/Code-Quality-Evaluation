class Solution:
    def minimumLength(self, s: str) -> int:
        # Explanation:
        # The allowed operation removes two occurrences of the same letter --
        # one immediately to the left and one immediately to the right of a chosen “middle”
        # occurrence. In other words, if you look at the positions where a given letter x
        # appears (say indices i0, i1, i2, …, i_{f-1}), an operation can be done only when
        # there is a “middle” element (that is, an occurrence that is not the first or last).
        # Such an operation will remove two occurrences from the group.
        # Bearing in mind that we do NOT remove the “middle” one, it turns out that for each letter:
        # •  If the letter appears 0 or 1 time, it stays as is.
        # •  If it appears 2 times, we cannot perform any valid operation (since neither occurrence is flanked on both sides),
        #    so both must remain.
        # •  If it appears 3 times, we can remove the two occurrences adjacent to the middle occurrence,
        #    leaving exactly one occurrence.
        # •  For frequencies 4, 5, etc., one can show by a simple simulation that the best we can do is:
        #       - For an even frequency f (f >= 2), we can remove pairs repeatedly until two remain.
        #       - For an odd frequency f (f >= 3), we can remove pairs repeatedly until one remains.
        # Hence, for each letter, if f is even and f >= 2, the best we can leave is 2; if f is odd (or f==1), the best
        # we can leave is 1.
        #
        # Since the operations affect occurrences only within the same letter (because the deletion is by comparing
        # equal characters), the process for each letter is independent. Thus, the overall answer is simply the
        # sum over all letters of the minimum surviving count for that letter.
        
        from collections import Counter
        counter = Counter(s)
        ans = 0
        for f in counter.values():
            if f < 2:
                ans += f
            else:
                # If frequency is even, minimum remains is 2; if odd, it's 1.
                ans += 1 if f % 2 == 1 else 2
        return ans

# For quick testing:
if __name__ == '__main__':
    sol = Solution()
    print(sol.minimumLength("abaacbcbb"))  # Expected output: 5
    print(sol.minimumLength("aa"))         # Expected output: 2