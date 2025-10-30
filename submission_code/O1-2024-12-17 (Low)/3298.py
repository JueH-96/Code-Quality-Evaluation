class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        """
        We want the longest "consecutive when sorted" subset from nums after optionally adding +1 to
        some elements (each element can be incremented at most once). Equivalently, each original
        element x can become either x or x+1. We then select a subset of these final values that
        forms the longest run of consecutive integers.
        
        A known (and somewhat subtle) way to solve this uses a "sliding window" over the possible
        final integer values and a careful bookkeeping of how many items we have at each integer
        (unincremented) or can move to the next integer (incremented). The main constraint is that
        one physical item cannot be used twice (i.e., we cannot increment the same element multiple
        times).

        The high-level idea of the implementation here is:
          1) Count how many of each integer we have in "freq[x]".
          2) We'll try to maintain a window of consecutive integers [left .. right].
             For each integer i in that range, we must use exactly one distinct item
             that is originally either i or i-1 (i.e., freq[i] or freq[i-1]) to "cover" i.
          3) If at some point we cannot cover right (because there are not enough unused items
             in freq[right] + freq[right-1]), we shrink from the left until we can cover right again.
          4) Track the maximum size of such a window.
        
        Implementation details:
         - We'll use two arrays (or dictionaries) to track how many items of each integer remain
           un-used: freq[x], and how many we have "shifted" from x to x+1: shift[x].
           But a more direct trick is to do incremental usage on the fly without needing shift[].
         - Because values can be up to 10^6, we'll allocate an array of that size + 2.
         - We'll sort, build freq[], and then do two-pointer over the integer range [1 .. max_val+1].
        
        The code below carefully implements a two-pointer window:
          - We keep a running count of how many items remain un-used in freq[i].
            At each new "right," we try to use exactly 1 item from freq[right] if possible;
            if freq[right] is exhausted, we try freq[right-1] (which means we're incrementing
            one of the original (right-1) items to cover "right").
          - If neither freq[right] nor freq[right-1] can supply an item, we must move "left" forward
            (and "free up" an item for the integer left was covering).
        
        This correctly enforces the "each element can be incremented at most once" rule and finds
        the longest run of consecutive integers we can fill.
        """

        from collections import Counter

        n = len(nums)
        if n == 1:
            return 1  # With a single element, answer is always 1.

        # Build frequency array up to max(nums)+1 (we might increment up to x+1)
        max_val = max(nums)
        freq = [0] * (max_val + 2)
        for x in nums:
            freq[x] += 1

        left = 1
        used_left = []  # Stack to remember from which freq[] we took for covering 'left'
        current_covered = 0
        answer = 0

        # Try to expand [left .. right]
        for right in range(1, max_val + 2):
            # Attempt to cover 'right' using an unused item from freq[right], else freq[right-1].
            covered = False
            if freq[right] > 0:
                # Use one item that is originally 'right'
                freq[right] -= 1
                used_left.append((right, 1))  # We used 1 from freq[right]
                current_covered += 1
                covered = True
            else:
                # Try to use an item originally (right-1), if available:
                if right - 1 >= 1 and freq[right - 1] > 0:
                    freq[right - 1] -= 1
                    used_left.append((right - 1, 1))  # We used 1 from freq[right-1] (incremented)
                    current_covered += 1
                    covered = True

            while not covered and left <= right:
                # We cannot cover 'right' now, so shrink from the left
                # "Free up" the item that was used at 'left'
                if used_left:
                    left_val, cnt = used_left.pop(0)
                    # That item was used to cover 'left', so put it back.
                    freq[left_val] += cnt
                    current_covered -= 1
                left += 1

                # After freeing the cover for 'left', the former 'left+1' now is the left boundary,
                # so 'left' is effectively uncovered. If left > right, break.
                if left > right:
                    break

                # Now try again to cover 'right'
                if freq[right] > 0:
                    freq[right] -= 1
                    used_left.append((right, 1))
                    current_covered += 1
                    covered = True
                else:
                    if right - 1 >= 1 and freq[right - 1] > 0:
                        freq[right - 1] -= 1
                        used_left.append((right - 1, 1))
                        current_covered += 1
                        covered = True

            if not covered:
                # Could not cover 'right' at all -> we are done (any larger 'right' won't
                # be coverable either in a strictly consecutive manner).
                break

            # Now [left .. right] is consecutive and covered
            answer = max(answer, right - left + 1)

        return answer