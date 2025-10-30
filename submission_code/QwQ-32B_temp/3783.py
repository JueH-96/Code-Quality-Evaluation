class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        if n == 0:
            return []
        
        # Precompute factorials up to n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i
        
        evens = []
        odds = []
        for num in range(1, n + 1):
            if num % 2 == 0:
                evens.append(num)
            else:
                odds.append(num)
        evens.sort()
        odds.sort()
        
        remaining_evens = evens.copy()
        remaining_odds = odds.copy()
        result = []
        current_k = k
        
        # Handle the first element
        candidates = sorted(evens + odds)
        found = False
        for candidate in candidates:
            if current_k == 0:
                break
            new_evens = remaining_evens.copy()
            new_odds = remaining_odds.copy()
            if candidate % 2 == 0:
                new_evens.remove(candidate)
                rem_evens = len(new_evens)
                rem_odds = len(new_odds)
                P = 'even'
            else:
                new_odds.remove(candidate)
                rem_evens = len(new_evens)
                rem_odds = len(new_odds)
                P = 'odd'
            
            # Calculate required even and odd counts for remaining positions
            remaining_positions = n - 1
            required_even = 0
            required_odd = 0
            current_parity = 'even' if P == 'odd' else 'odd'
            for _ in range(remaining_positions):
                if current_parity == 'even':
                    required_even += 1
                else:
                    required_odd += 1
                current_parity = 'even' if current_parity == 'odd' else 'odd'
            
            if rem_evens == required_even and rem_odds == required_odd:
                perm_count = fact[rem_evens] * fact[rem_odds]
                if current_k <= perm_count:
                    result.append(candidate)
                    if candidate % 2 == 0:
                        remaining_evens = new_evens
                    else:
                        remaining_odds = new_odds
                    start_parity = P
                    current_parity = 'even' if start_parity == 'odd' else 'odd'
                    found = True
                    break
                else:
                    current_k -= perm_count
            else:
                continue
        
        if not found:
            return []
        
        # Handle remaining positions
        for i in range(1, n):
            required_parity = current_parity
            if required_parity == 'even':
                candidates = remaining_evens.copy()
            else:
                candidates = remaining_odds.copy()
            candidates.sort()
            found_next = False
            for candidate in candidates:
                new_evens = remaining_evens.copy()
                new_odds = remaining_odds.copy()
                if required_parity == 'even':
                    new_evens.remove(candidate)
                    rem_evens = len(new_evens)
                    rem_odds = len(new_odds)
                else:
                    new_odds.remove(candidate)
                    rem_evens = len(new_evens)
                    rem_odds = len(new_odds)
                
                remaining_positions = n - len(result) - 1
                required_even = 0
                required_odd = 0
                current_p = current_parity
                for _ in range(remaining_positions):
                    if current_p == 'even':
                        required_even += 1
                    else:
                        required_odd += 1
                    current_p = 'even' if current_p == 'odd' else 'odd'
                
                if rem_evens == required_even and rem_odds == required_odd:
                    perm_count = fact[rem_evens] * fact[rem_odds]
                    if current_k <= perm_count:
                        result.append(candidate)
                        if required_parity == 'even':
                            remaining_evens = new_evens
                        else:
                            remaining_odds = new_odds
                        current_parity = 'even' if current_parity == 'odd' else 'odd'
                        found_next = True
                        break
                    else:
                        current_k -= perm_count
                else:
                    continue
            if not found_next:
                return []
        
        return result