import math
from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        def count_remaining_perms(available, pattern):
            if not pattern:
                return 1
            
            odds_available = len([x for x in available if x % 2 == 1])
            evens_available = len(available) - odds_available
            odds_needed = sum(pattern)
            evens_needed = len(pattern) - odds_needed
            
            if odds_needed == odds_available and evens_needed == evens_available:
                return math.factorial(odds_available) * math.factorial(evens_available)
            else:
                return 0
        
        def find_kth_alternating(available, pattern, k):
            if not pattern:
                return []
            
            if len(available) == 1:
                return available
            
            current_needs_odd = pattern[0]
            remaining_pattern = pattern[1:]
            
            candidates = []
            for num in available:
                if (num % 2 == 1) == current_needs_odd:
                    candidates.append(num)
            
            candidates.sort()
            
            for candidate in candidates:
                remaining_available = [x for x in available if x != candidate]
                count = count_remaining_perms(remaining_available, remaining_pattern)
                
                if k <= count:
                    return [candidate] + find_kth_alternating(remaining_available, remaining_pattern, k)
                else:
                    k -= count
            
            return []
        
        available = list(range(1, n+1))
        
        # Generate possible starting patterns
        patterns = []
        if n % 2 == 1:
            # Only pattern starting with odd: O-E-O-E...O
            patterns.append([i % 2 == 0 for i in range(n)])
        else:
            # Two patterns: O-E-O-E... and E-O-E-O...
            patterns.append([i % 2 == 0 for i in range(n)])  # O-E-O-E...
            patterns.append([i % 2 == 1 for i in range(n)])  # E-O-E-O...
        
        # Process all starting numbers in lexicographic order
        for first_num in sorted(available):
            for pattern in patterns:
                if (first_num % 2 == 1) == pattern[0]:
                    # This pattern can start with first_num
                    remaining_available = [x for x in available if x != first_num]
                    remaining_pattern = pattern[1:]
                    count = count_remaining_perms(remaining_available, remaining_pattern)
                    
                    if k <= count:
                        return [first_num] + find_kth_alternating(remaining_available, remaining_pattern, k)
                    else:
                        k -= count
        
        return []