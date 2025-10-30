class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        result = []
        
        for a, b, c, d in queries:
            can_make = True
            
            # Track what characters are needed in each rearrangeable region
            first_half_needed = {}
            second_half_needed = {}
            
            # Check each position in the first half
            for i in range(n // 2):
                mirror = n - 1 - i
                
                in_first_range = a <= i <= b
                in_second_range = c <= mirror <= d
                
                if in_first_range and in_second_range:
                    # Both can be rearranged - handle later
                    continue
                elif in_first_range and not in_second_range:
                    # First half position can be rearranged, second half is fixed
                    char = s[mirror]
                    first_half_needed[char] = first_half_needed.get(char, 0) + 1
                elif not in_first_range and in_second_range:
                    # First half position is fixed, second half can be rearranged
                    char = s[i]
                    second_half_needed[char] = second_half_needed.get(char, 0) + 1
                else:
                    # Both positions are fixed - must already match
                    if s[i] != s[mirror]:
                        can_make = False
                        break
            
            if not can_make:
                result.append(False)
                continue
            
            # Count available characters in each rearrangeable region
            first_half_available = {}
            for i in range(a, b + 1):
                char = s[i]
                first_half_available[char] = first_half_available.get(char, 0) + 1
            
            second_half_available = {}
            for i in range(c, d + 1):
                char = s[i]
                second_half_available[char] = second_half_available.get(char, 0) + 1
            
            # Check if we have enough characters to satisfy constraints
            for char, count in first_half_needed.items():
                if first_half_available.get(char, 0) < count:
                    can_make = False
                    break
            
            if not can_make:
                result.append(False)
                continue
            
            for char, count in second_half_needed.items():
                if second_half_available.get(char, 0) < count:
                    can_make = False
                    break
            
            if not can_make:
                result.append(False)
                continue
            
            # Remove used characters
            for char, count in first_half_needed.items():
                first_half_available[char] -= count
                if first_half_available[char] == 0:
                    del first_half_available[char]
            
            for char, count in second_half_needed.items():
                second_half_available[char] -= count
                if second_half_available[char] == 0:
                    del second_half_available[char]
            
            # Remaining characters must match exactly
            result.append(first_half_available == second_half_available)
        
        return result