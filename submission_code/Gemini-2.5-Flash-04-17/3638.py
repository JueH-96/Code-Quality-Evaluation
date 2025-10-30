from collections import Counter
import math

class Solution:
    def makeStringGood(self, s: str) -> int:
        n = len(s)
        initial_counts = [0] * 26
        for char in s:
            initial_counts[ord(char) - ord('a')] += 1

        def calculate_cost(target_counts):
            """Calculates the minimum operations to transform initial_counts to target_counts.
               This model assumes changes only happen from c to c+1, costing 1 per step."""
            cost = 0
            carry = 0 # Represents characters conceptually moved from char i-1 to i

            for i in range(26):
                initial_at_char = initial_counts[i]
                needed_at_char = target_counts[i]

                available_at_char = initial_at_char + carry

                if available_at_char >= needed_at_char:
                    # Surplus at this character index.
                    # These surplus characters must be either deleted or converted
                    # one step forward (to i+1). Both operations cost 1 per character.
                    # The cost reflects the minimum operations to handle the surplus.
                    surplus = available_at_char - needed_at_char
                    cost += surplus
                    carry = surplus # These surplus characters are conceptually available for the next step
                else:
                    # Deficit at this character index.
                    # These deficit characters must be inserted.
                    deficit = needed_at_char - available_at_char
                    cost += deficit
                    carry = 0 # No surplus to carry over from this step

            # The final `carry` represents surplus characters at 'z' that must be deleted.
            # This cost was already added in the loop at i=25.
            # The total cost is the sum of costs at each step.
            return cost

        # Option 1: delete all characters (target is empty string, m=0, k=0).
        # This is a valid good string. The cost is n deletions.
        min_ops = n 

        # Iterate through possible number of distinct characters in the good string (m)
        # m=0 case is handled by min_ops = n
        for m in range(1, 27):
            # Iterate through possible starting character index of the contiguous block (s_idx)
            for s_idx in range(26 - m + 1):
                # The target characters are at indices s_idx, s_idx+1, ..., s_idx+m-1

                # For a fixed block [s_idx, s_idx+m-1] and fixed m,
                # the cost function is convex w.r.t. the target count k.
                # We can use ternary search to find the optimal integer k.

                # Determine a search range for k.
                # k must be non-negative.
                # If k is very large, say k > n + alphabet_size, the cost related to insertions will be large.
                # If k is very small, the cost related to deletions/conversions will be large.
                # A safe upper bound for k could be related to the total number of characters n,
                # plus a buffer for characters created by conversions/insertions.
                # The maximum possible value of a character index is 25 ('z').
                # If we convert n 'a's to 'z's, it takes cost n * 25.
                # If we then need k 'z's, and have n, we need k-n insertions.
                # Max possible count of any char can be up to n + sum of carries. Max carry <= n.
                # A reasonable upper bound for k is n + alphabet_size. Let's use n + 26.
                
                low_k = 0
                high_k = n + 26 # Search range for k [0, n+26] inclusive

                # Ternary search for integer k
                # The range [low_k, high_k] is inclusive.
                while high_k - low_k >= 3:
                    k1 = low_k + (high_k - low_k) // 3
                    k2 = high_k - (high_k - low_k) // 3
                    # Ensure k1 != k2, although // 3 handles this if high_k - low_k >= 3

                    # Construct target_counts for k1 and k2 based on the current block
                    target_counts1 = [0] * 26
                    target_counts2 = [0] * 26
                    for i in range(s_idx, s_idx + m):
                        target_counts1[i] = k1
                        target_counts2[i] = k2

                    # Calculate cost for k1 and k2
                    cost1 = calculate_cost(target_counts1)
                    cost2 = calculate_cost(target_counts2)

                    if cost1 < cost2:
                        # The minimum is likely to the left of k2.
                        # The search space becomes [low_k, k2].
                        high_k = k2 
                    else:
                        # The minimum is likely to the right of or at k1.
                        # The search space becomes [k1, high_k].
                        low_k = k1

                # After the loop, the minimum is in the small range [low_k, high_k].
                # Check all values in this remaining range.
                for k in range(low_k, high_k + 1):
                    target_counts = [0] * 26
                    for i in range(s_idx, s_idx + m):
                        target_counts[i] = k
                    min_ops = min(min_ops, calculate_cost(target_counts))

        return min_ops