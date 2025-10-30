class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        result = []
        
        for a, b, c, d in queries:
            # Create a set of positions that can be rearranged
            rearrangeable = set()
            for i in range(a, b + 1):
                rearrangeable.add(i)
            for i in range(c, d + 1):
                rearrangeable.add(i)
            
            # Check if we can make a palindrome
            can_make = True
            
            # First check: positions that cannot be rearranged must already match
            for i in range(n // 2):
                mirror = n - 1 - i
                if i not in rearrangeable and mirror not in rearrangeable:
                    if s[i] != s[mirror]:
                        can_make = False
                        break
            
            if can_make:
                # Second check: collect all characters from rearrangeable positions
                # and their mirrors, check if they can form palindrome pairs
                char_count = {}
                
                # Count characters from left half that are rearrangeable or whose mirrors are
                for i in range(n // 2):
                    mirror = n - 1 - i
                    if i in rearrangeable or mirror in rearrangeable:
                        char_count[s[i]] = char_count.get(s[i], 0) + 1
                        char_count[s[mirror]] = char_count.get(s[mirror], 0) + 1
                
                # Check if all characters appear in even counts
                for count in char_count.values():
                    if count % 2 != 0:
                        can_make = False
                        break
            
            result.append(can_make)
        
        return result