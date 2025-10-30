class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)

        # The minimum possible value for the result (min length of longest substring) is 1.
        # The maximum possible value is n (when numOps is 0 and string is homogeneous).
        low = 1
        high = n
        ans = n # Initialize with the maximum possible answer

        # Binary search for the minimum k (the minimum possible value for the maximum length)
        while low <= high:
            mid = (low + high) // 2
            
            # Check if it's possible to make the longest homogeneous substring
            # have length at most `mid` using at most `numOps` flips.
            k = mid

            # Calculate minimum flips (0->1) needed to ensure no '0' segment is longer than k
            # This is done by iterating through the string and counting consecutive '0's.
            # If the count exceeds k, we must perform a flip (0->1) to break the run.
            # The greedy strategy is to flip the last '0' in the run, effectively
            # resetting the current consecutive '0' count.
            flips0 = 0
            current_zeros = 0
            for char in s:
                if char == '0':
                    current_zeros += 1
                else:
                    current_zeros = 0 # Run of '0's is broken by a '1'
                if current_zeros > k:
                    # Found k+1 consecutive '0's. Need to break this run.
                    # Flip the current character (which is '0').
                    # This costs 1 flip and ends the current '0' run at this position.
                    flips0 += 1
                    current_zeros = 0 # The character at this position is now conceptually a '1'

            # If flips needed for '0's alone exceed the total allowed flips,
            # then achieving max length k is impossible.
            if flips0 > numOps:
                low = mid + 1 # Need a larger max length
                continue

            # Calculate minimum flips (1->0) needed to ensure no '1' segment is longer than k
            # Similar greedy approach as for '0's.
            flips1 = 0
            current_ones = 0
            for char in s:
                if char == '1':
                    current_ones += 1
                else:
                    current_ones = 0 # Run of '1's is broken by a '0'
                if current_ones > k:
                    # Found k+1 consecutive '1's. Need to break this run.
                    # Flip the current character (which is '1').
                    # This costs 1 flip and ends the current '1' run at this position.
                    flips1 += 1
                    current_ones = 0 # The character at this position is now conceptually a '0'

            # The minimum total flips needed to satisfy both conditions simultaneously
            # is the sum of flips needed for each condition independently,
            # because the required flips are of different types (0->1 vs 1->0)
            # and they address independent constraints (max length of '0' runs vs '1' runs).
            if (flips0 + flips1) <= numOps:
                # If we can achieve max length 'mid' using <= numOps flips,
                # then 'mid' is a possible answer. We try to find a smaller possible answer.
                ans = mid
                high = mid - 1
            else:
                # Cannot achieve max length 'mid' with the given numOps.
                # Need a larger max length.
                low = mid + 1

        return ans