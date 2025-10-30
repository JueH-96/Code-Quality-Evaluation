class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10 ** 9 + 7
        
        # Create a dictionary to store the count of each suffix
        suffix_count = {}
        for i in range(n):
            suffix = s[i:]
            suffix_count[suffix] = suffix_count.get(suffix, 0) + 1
        
        # Iterate through the string t and count the number of ways to transform s into t
        ways = 0
        for i in range(n):
            prefix = t[:i+1]
            if prefix in suffix_count:
                ways = (ways + suffix_count[prefix]) % MOD
        
        # Return the number of ways if k is 1, otherwise return 0
        return ways if k == 1 else 0