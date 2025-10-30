class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        # If the number is already "0", it's special.
        if num == "0":
            return 0

        # Valid target endings for a number divisible by 25.
        candidates = ["00", "25", "50", "75"]
        best = float('inf')
        
        # Try to achieve one of the valid endings by deleting other digits.
        for target in candidates:
            # We need to fix the last two digits of final number to be target.
            # We'll search from the end for target[1] (the last digit)
            pos2 = -1
            for i in range(n - 1, -1, -1):
                if num[i] == target[1]:
                    pos2 = i
                    break
            if pos2 == -1:
                continue  # target[1] not found; cannot form this candidate
            
            # Now search for target[0] to the left of pos2
            pos1 = -1
            for i in range(pos2 - 1, -1, -1):
                if num[i] == target[0]:
                    pos1 = i
                    break
            if pos1 == -1:
                continue  # target[0] not found on the left side; cannot form this candidate
            
            # Calculate the number of deletions needed.
            # Any digits after pos2 must be removed.
            deletions_after_pos2 = n - pos2 - 1
            # Any digits between pos1 and pos2 that are not required must be removed.
            deletions_between = pos2 - pos1 - 1
            total_deletions = deletions_after_pos2 + deletions_between
            best = min(best, total_deletions)
        
        # If we found a valid candidate pair, best holds the minimal deletions needed.
        # However, there's also the possibility of making the number "0" (which is divisible by 25)
        # by deleting all digits except a single '0'. 
        has_zero = '0' in num
        # If there is at least one '0' in num, we can delete all digits except one that is '0'
        # to get "0". That would cost (n - 1) operations.
        candidate_zero = n - 1 if has_zero else n

        # The overall answer is the minimum between our candidate pair result and candidate_zero.
        return int(min(best, candidate_zero))