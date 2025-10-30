class Solution:
    def minimumOperations(self, num: str) -> int:
        targets = ["00", "25", "50", "75"]
        n = len(num)
        res = float("inf")
        
        # For each candidate two-digit ending,
        # try to find minimal deletions such that the last two digits (in order) form the candidate.
        for t in targets:
            # We try to match t[1] then t[0] scanning from right.
            j = n - 1  # pointer to search in num from right side
            ops = 0
            found = True
            # We want to form t[1] then t[0] in that order.
            for digit in reversed(t):
                # find digit from right pointer j.
                while j >= 0 and num[j] != digit:
                    ops += 1
                    j -= 1
                if j < 0:
                    found = False
                    break
                # Found the digit, move left to look for the next digit.
                j -= 1
            if found:
                res = min(res, ops)
                
        return res if res != float("inf") else 0