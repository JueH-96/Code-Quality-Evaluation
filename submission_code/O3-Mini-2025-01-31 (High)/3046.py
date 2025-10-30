class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        # Candidate pairs (as strings) whose integer value mod 25 is 0.
        # Note: For any integer to be divisible by 25 (except 0 itself),
        # its last two digits must be 00, 25, 50, or 75.
        candidates = ["00", "25", "50", "75"]
        ans = float("inf")
        
        # Try to form a valid ending (with two digits) for each candidate pair.
        for pair in candidates:
            # We will work backwards and count how many digits we "skip" (i.e. delete)
            # to have these two digits as the ending.
            currentOps = 0
            i = n - 1  # start scanning from the end
            
            # First, find the second digit (the last digit of the desired pair).
            target_second = pair[1]
            while i >= 0 and num[i] != target_second:
                currentOps += 1
                i -= 1
            if i < 0:
                # Candidate not possible because we didn't find the required digit.
                continue
            
            # Now, find the first digit of the pair (which must come before the second digit).
            i -= 1  # move left one position
            target_first = pair[0]
            while i >= 0 and num[i] != target_first:
                currentOps += 1
                i -= 1
            if i < 0:
                # Candidate not possible.
                continue
            
            ans = min(ans, currentOps)
        
        # If none of the candidate pairs worked, we must consider transforming num to 0.
        # We are allowed to delete digits arbitrarily. Note that if we delete all digits,
        # the problem statement says that num becomes 0 (which is special).
        # However, if there is at least one '0' in the number,
        # we can get the number "0" by deleting all digits except that one '0'.
        # That would cost n-1 deletions.
        if ans == float("inf"):
            if "0" in num:
                return n - 1  # keep one '0', delete the rest
            else:
                # If there is no '0' then the only option is to delete all digits.
                return n
        
        return ans