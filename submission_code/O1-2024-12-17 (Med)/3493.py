class Solution:
    def maxOperations(self, s: str) -> int:
        #
        # Idea:
        #
        # Each operation in the problem corresponds (in a "bubble‐sort" sense) to flipping
        # a "10" → "01".  However, unlike single‐step adjacent swaps, here a '1' may jump
        # over several zeros in one go, counting as only one operation.  We want the
        # MAXIMUM number of such operations.
        #
        # A useful way to see this is:
        #   1) Count how many (1,0) "inversions" there are in total, i.e. for every '1'
        #      how many '0's lie to its right.  If each crossing over one zero were a
        #      separate operation, we would get exactly that many operations.
        #   2) However, when a '1' jumps over a block of consecutive zeros in one step,
        #      that effectively compresses what would have been multiple single‐zero
        #      swaps into one operation.  So we need to subtract the "extra" merges from
        #      the naive inversion count.
        #
        # Crucially, in the final arrangement, all 1s remain in their relative order
        # (they do not pass one another).  Each '1' effectively jumps over consecutive
        # zeros lying immediately between it and the next '1' to its right (if any).  So
        # for each gap of length G>0 of consecutive zeros between consecutive 1s, that
        # gap is jumped in a single operation by exactly that left '1'.  If G>1, that
        # merges G single‐zero crossings into 1 operation, so we reduce our count by
        # (G - 1).
        #
        # Steps to implement efficiently:
        #
        #   (A) Count "inversions":
        #       - Let total_zeros = number of '0' in s.
        #       - Traverse s left to right:
        #           if we see '1',  add current total_zeros to "inversions"
        #           if we see '0',  decrement total_zeros by 1
        #       The result is the total (1,0) inversions.
        #
        #   (B) Track positions of the '1's in an array "ones".
        #       For each consecutive pair of 1s at positions ones[i], ones[i+1]:
        #         gap = ones[i+1] - ones[i] - 1   (number of zeros between them)
        #         if gap > 0, we subtract (gap - 1) from the answer, because
        #         jumping across gap zeros in one move merges gap single‐zero crosses into 1.
        #
        #   (C) The final number = (inversions) - sum( (gap - 1) for each gap>0 ).
        #
        # This matches all the given examples and runs in O(n).
        #

        n = len(s)
        zero_count = s.count('0')

        # (A) Compute total inversions
        inversions = 0
        z = zero_count
        for ch in s:
            if ch == '1':
                inversions += z
            else:
                # we've now passed one zero
                z -= 1

        # (B) Gather positions of '1'
        ones = []
        for i, ch in enumerate(s):
            if ch == '1':
                ones.append(i)

        # If there are 0 or 1 ones, no "10" operations are possible anyway.
        if len(ones) <= 1:
            return 0

        # (C) Subtract merged "jumps" for consecutive 1s
        answer = inversions
        for i in range(len(ones) - 1):
            gap = ones[i+1] - ones[i] - 1  # number of zeros between consecutive 1s
            if gap > 0:
                # jumping across these 'gap' zeros in one step merges what would have
                # been 'gap' single-step swaps into 1 operation => reduce (gap - 1)
                answer -= (gap - 1)

        return answer