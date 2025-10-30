class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations
        
        # Generate all possible combinations of the three strings
        perms = [''.join(p) for p in permutations([a, b, c])]
        
        # Find the lexicographically smallest string that contains all three strings as substrings
        for perm in perms:
            if all(s in perm for s in [a, b, c]):
                return perm
        
        # If no such string is found, return an empty string
        return ""