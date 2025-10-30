from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Precompute factorials up to n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i
        
        # Separate the numbers into odds and evens
        odds = [i for i in range(1, n+1) if i % 2 == 1]
        evens = [i for i in range(1, n+1) if i % 2 == 0]
        o = len(odds)
        e = len(evens)
        
        # Calculate total number of valid permutations
        if n % 2 == 0:
            total = 2 * fact[o] * fact[e]
        else:
            total = fact[o] * fact[e]
        
        # Check if k is out of range
        if k > total:
            return []
        
        start_with_odd = True
        current_k = k
        
        # Determine the starting group for even n
        if n % 2 == 0:
            half = fact[o] * fact[e]
            if current_k > half:
                start_with_odd = False
                current_k -= half
        
        # Initialize remaining odds and evens
        remaining_odds = list(odds)
        remaining_evens = list(evens)
        
        result = []
        
        # Generate the permutation step by step
        for i in range(n):
            if start_with_odd:
                required_parity_odd = (i % 2 == 0)
            else:
                required_parity_odd = (i % 2 == 1)
            
            if required_parity_odd:
                # Need to choose from remaining_odds
                found = False
                for j in range(len(remaining_odds)):
                    current_odds_len = len(remaining_odds)
                    current_evens_len = len(remaining_evens)
                    num = fact[current_odds_len - 1] * fact[current_evens_len]
                    if current_k > num:
                        current_k -= num
                        continue
                    else:
                        result.append(remaining_odds[j])
                        del remaining_odds[j]
                        found = True
                        break
                if not found:
                    return []
            else:
                # Need to choose from remaining_evens
                found = False
                for j in range(len(remaining_evens)):
                    current_evens_len = len(remaining_evens)
                    current_odds_len = len(remaining_odds)
                    num = fact[current_odds_len] * fact[current_evens_len - 1]
                    if current_k > num:
                        current_k -= num
                        continue
                    else:
                        result.append(remaining_evens[j])
                        del remaining_evens[j]
                        found = True
                        break
                if not found:
                    return []
        
        return result