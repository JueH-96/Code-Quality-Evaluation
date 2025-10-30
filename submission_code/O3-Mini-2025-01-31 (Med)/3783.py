from typing import List
import sys
sys.setrecursionlimit(10000)

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # We'll use DP memoization to count the number of valid completions from a given state.
        # State is (odd_count, even_count, last_parity)
        # last_parity: -1 indicates no previous element (None), 0 for even, 1 for odd.
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(odd_count: int, even_count: int, last_parity: int) -> int:
            # Base: no numbers left to place.
            if odd_count == 0 and even_count == 0:
                return 1
            total = 0
            if last_parity == -1:
                # No restriction; we can choose any available number.
                if odd_count > 0:
                    # When we choose any odd, the remaining count is: odd_count - 1, even_count and now last_parity becomes 1.
                    total += odd_count * dp(odd_count - 1, even_count, 1)
                if even_count > 0:
                    total += even_count * dp(odd_count, even_count - 1, 0)
            elif last_parity == 0:
                # last was even -> must choose odd.
                if odd_count > 0:
                    total += odd_count * dp(odd_count - 1, even_count, 1)
            elif last_parity == 1:
                # last was odd -> must choose even.
                if even_count > 0:
                    total += even_count * dp(odd_count, even_count - 1, 0)
            return total
        
        # Prepare the sorted lists of odds and evens.
        odds = [i for i in range(1, n+1) if i % 2 == 1]
        evens = [i for i in range(1, n+1) if i % 2 == 0]
        # total count from the start state:
        total_count = dp(len(odds), len(evens), -1)
        if k > total_count:  # not enough valid permutations.
            return []
        
        result = []
        # The last chosen parity: -1 means none, 0 means even, 1 means odd.
        last_parity = -1
        
        # Iteratively choose next element
        # The state is given by current sorted lists (odds, evens) and last_parity.
        while odds or evens:
            # Determine allowed candidates based on parity condition.
            candidates = []
            if last_parity == -1:
                # allowed: both odds and evens.
                # Merge the two lists while keeping them sorted.
                i, j = 0, 0
                while i < len(odds) or j < len(evens):
                    if j >= len(evens) or (i < len(odds) and odds[i] < evens[j]):
                        candidates.append((odds[i], 1))  # 1 indicates odd.
                        i += 1
                    else:
                        candidates.append((evens[j], 0))  # 0 indicates even.
                        j += 1
            elif last_parity == 0:
                # last was even, so must choose odd.
                # candidates come from odds list (which is sorted).
                for num in odds:
                    candidates.append((num, 1))
            else:  # last_parity == 1
                # last was odd, so must choose even.
                for num in evens:
                    candidates.append((num, 0))
            
            chosen = False
            for idx, (num, parity) in enumerate(candidates):
                # For each candidate, calculate the count of completions if we choose this candidate.
                if parity == 1:
                    # candidate is odd: remove one odd from odds.
                    cnt = dp(len(odds)-1, len(evens), 1)
                else:  # parity == 0, candidate is even.
                    cnt = dp(len(odds), len(evens)-1, 0)
                
                if k > cnt:
                    k -= cnt
                else:
                    # We choose this candidate.
                    result.append(num)
                    if parity == 1:
                        # Remove the candidate number from odds.
                        # Since odds is sorted and candidate came from odds,
                        # find and remove the specific number.
                        odds.remove(num)
                    else:
                        evens.remove(num)
                    # Update last parity to the parity of the chosen number.
                    last_parity = parity
                    chosen = True
                    break
            
            if not chosen:  # If no candidate is chosen, something went wrong.
                return []
        return result