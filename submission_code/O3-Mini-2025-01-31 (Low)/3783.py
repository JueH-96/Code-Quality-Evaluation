from math import factorial
from bisect import bisect_left
from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # We are to generate the k-th lexicographically sorted permutation
        # of [1,2,...,n] that is "alternating" i.e. consecutive numbers must have different parity.
        #
        # Observation:
        # Any valid alternating permutation must alternate between odd and even.
        # But note that the parity that occurs more often (if n is odd, odds occur one more time)
        # must be placed in the first slot.
        # In short:
        #   If n is odd, the permutation MUST start with an odd number.
        #   If n is even, both possibilities (starting with odd or even) are allowed.
        #
        # Once the first number is chosen, the remainder's parity is forced:
        #   If first number is odd, then the pattern becomes odd, even, odd, even, ... and vice versa.
        #
        # Now, consider that once the pattern (i.e. required parity at each position)
        # is fixed, the positions which demand an odd number can be filled with any permutation
        # of the available odd numbers, and similarly for the even numbers.
        # Therefore, if at some moment we have r odds and s evens left to place (and the next
        # required parity is fixed), then the number of completions is simply:
        #     completions = (r)! * (s)!   (where the factorial is for the ordering among those positions).
        #
        # We can then “build” the k-th valid permutation digit by digit:
        #   - At each position, iterate through the candidate numbers (from the available set)
        #     that satisfy the parity requirement.
        #   - For each candidate, compute how many completions exist if we choose that candidate.
        #   - If the count is >= k, we pick that candidate and move on.
        #   - Otherwise, subtract that count from k and try the next candidate.
        #
        # One subtle point: for the very first position (pos==0)
        # when n is even, both odd and even are allowed, but we must consider the lexicographical order.
        # This means that we iterate all candidates from the overall sorted list.
        #
        # We use precomputed factorial values up to n.
        
        # Precompute factorials up to n (n <= 100)
        fact = [1] * (n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1] * i

        # Build sorted lists for odd and even numbers from 1..n.
        odds_all = [x for x in range(1, n+1) if x % 2 == 1]
        evens_all = [x for x in range(1, n+1) if x % 2 == 0]
        
        # We'll use lists to represent the available numbers.
        # For easy removal in lex order we keep them sorted.
        avail_odds = odds_all.copy()
        avail_evens = evens_all.copy()
        
        # A helper function that given remaining counts, returns number of completions.
        # Here, it doesn’t matter what the next required parity is, since the positions are fixed:
        # The remaining odds can only go to odd slots and evens only to even slots.
        def completions(odds_left, evens_left):
            return fact[odds_left] * fact[evens_left]
        
        # Result permutation:
        permutation = []
        
        # We'll be building the permutation digit by digit.
        # For the first digit, candidate must follow allowed parity.
        # If n is odd, must be odd. If n is even, any number (both odd and even) is allowed.
        #
        # But note: the pattern is determined by the first digit.
        # If we choose an odd first digit, then the remainder must alternate starting with even.
        # If we choose an even first digit, then the remainder must alternate starting with odd.
        #
        # Let curr_parity represent the parity required at the current position.
        # At pos 0, if n is even, candidates of both parities are possible.
        # So we treat pos0 separately.
        
        total_positions = n
        
        # We will get the answer by iterating over positions.
        pos = 0
        
        # For pos 0, the candidate set depends on n:
        if n % 2 == 1:
            # Must choose an odd number
            candidates = avail_odds  # sorted list
            # For each candidate, if chosen then we remove it from avail_odds.
            for candidate in candidates:
                # If we choose candidate (which is odd), then
                # new avail_odds count becomes len(avail_odds)-1, and avail_evens remains the same.
                # And the next required parity becomes even.
                cnt = completions(len(avail_odds)-1, len(avail_evens))
                if cnt < k:
                    k -= cnt
                else:
                    # Choose candidate.
                    permutation.append(candidate)
                    # Remove candidate from avail_odds.
                    avail_odds.remove(candidate)
                    # Set next required parity:
                    next_parity = 0  # 0 stands for even.
                    pos += 1
                    break
            else:
                # If loop finishes without break, there are not enough permutations.
                return []
        else:
            # n is even; first position: both odd and even allowed.
            # We'll iterate over all available numbers from 1..n in sorted order.
            # For each candidate, determine what parity it is and then count completions.
            # If candidate is odd: then if we pick it, avail_odds decreases, and next parity required is even.
            # If candidate is even: then avail_evens decreases and next required parity required is odd.
            merged = sorted(avail_odds + avail_evens)
            chosen = None
            next_parity = None
            for candidate in merged:
                if candidate % 2 == 1:
                    cnt = completions(len(avail_odds)-1, len(avail_evens))
                else:
                    cnt = completions(len(avail_odds), len(avail_evens)-1)
                if cnt < k:
                    k -= cnt
                else:
                    chosen = candidate
                    permutation.append(candidate)
                    if candidate % 2 == 1:
                        avail_odds.remove(candidate)
                        next_parity = 0  # next must be even since candidate is odd.
                    else:
                        avail_evens.remove(candidate)
                        next_parity = 1  # next must be odd since candidate is even.
                    pos += 1
                    break
            if chosen is None:
                return []
        
        # Now, for positions 1 to n-1, the next required parity is fixed (stored in next_parity).
        # For each subsequent position, we restrict candidate to available numbers of that parity.
        while pos < total_positions:
            if next_parity == 1:
                # Need to choose an odd number.
                candidates = avail_odds.copy()  # sorted
            else:
                candidates = avail_evens.copy()  # sorted
            if not candidates:
                # No candidate available but permutation not complete.
                return []
            chosen = None
            for candidate in candidates:
                # Determine new available counts after picking candidate.
                if candidate % 2 == 1:
                    new_odds = len(avail_odds) - 1
                    new_evens = len(avail_evens)
                    new_required = 0  # after odd, require even.
                else:
                    new_odds = len(avail_odds)
                    new_evens = len(avail_evens) - 1
                    new_required = 1  # after even, require odd.
                cnt = completions(new_odds, new_evens)
                if cnt < k:
                    k -= cnt
                else:
                    chosen = candidate
                    permutation.append(candidate)
                    if candidate % 2 == 1:
                        avail_odds.remove(candidate)
                    else:
                        avail_evens.remove(candidate)
                    next_parity = new_required
                    pos += 1
                    break
            if chosen is None:
                # If none candidate yielded enough completions, then k was too high.
                return []
        return permutation