class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        def factorial(x):
            if x <= 1:
                return 1
            result = 1
            for i in range(2, x + 1):
                result *= i
            return result
        
        def count_alternating_perms(odds_left, evens_left, need_odd):
            if odds_left == 0 and evens_left == 0:
                return 1
            if need_odd and odds_left == 0:
                return 0
            if not need_odd and evens_left == 0:
                return 0
            
            if need_odd:
                return factorial(odds_left) * factorial(evens_left)
            else:
                return factorial(odds_left) * factorial(evens_left)
        
        # Separate odd and even numbers
        odds = [i for i in range(1, n + 1) if i % 2 == 1]
        evens = [i for i in range(1, n + 1) if i % 2 == 0]
        
        # Check if we have enough numbers to alternate
        if abs(len(odds) - len(evens)) > 1:
            return []
        
        # Calculate total valid permutations
        total_perms = 0
        if len(odds) == len(evens):
            # Can start with either odd or even
            total_perms = 2 * factorial(len(odds)) * factorial(len(evens))
        elif len(odds) == len(evens) + 1:
            # Must start with odd
            total_perms = factorial(len(odds)) * factorial(len(evens))
        else:
            # Must start with even
            total_perms = factorial(len(odds)) * factorial(len(evens))
        
        if k > total_perms:
            return []
        
        result = []
        remaining_odds = odds[:]
        remaining_evens = evens[:]
        k -= 1  # Convert to 0-indexed
        
        # Determine if we start with odd or even
        need_odd = True
        if len(odds) == len(evens):
            # Can start with either
            perms_starting_odd = factorial(len(odds)) * factorial(len(evens))
            if k >= perms_starting_odd:
                need_odd = False
                k -= perms_starting_odd
        elif len(evens) > len(odds):
            need_odd = False
        
        # Build the permutation
        for pos in range(n):
            if need_odd:
                # Choose from remaining odds
                perms_per_choice = factorial(len(remaining_odds) - 1) * factorial(len(remaining_evens))
                choice_idx = k // perms_per_choice
                result.append(remaining_odds[choice_idx])
                remaining_odds.pop(choice_idx)
                k %= perms_per_choice
                need_odd = False
            else:
                # Choose from remaining evens
                perms_per_choice = factorial(len(remaining_odds)) * factorial(len(remaining_evens) - 1)
                choice_idx = k // perms_per_choice
                result.append(remaining_evens[choice_idx])
                remaining_evens.pop(choice_idx)
                k %= perms_per_choice
                need_odd = True
        
        return result