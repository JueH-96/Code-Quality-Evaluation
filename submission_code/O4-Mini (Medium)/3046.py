class Solution:
    def minimumOperations(self, num: str) -> int:
        L = len(num)
        # Possible endings for a number divisible by 25
        targets = [("0","0"), ("2","5"), ("5","0"), ("7","5")]
        ans = float('inf')
        
        # Try to form each target ending by deleting digits
        for a, b in targets:
            # j is position of b (the last digit)
            for j in range(L-1, -1, -1):
                if num[j] != b:
                    continue
                # i is position of a before j
                for i in range(j-1, -1, -1):
                    if num[i] == a:
                        # delete all digits between i and j, and all after j
                        ops = (j - i - 1) + (L - 1 - j)
                        ans = min(ans, ops)
                        break  # no need to look for earlier i for this j
        
        # Special case: leave a single '0' (resulting number "0" is divisible by 25)
        # Cost = delete all other digits = L-1 operations, if there's any '0'
        if '0' in num:
            ans = min(ans, L - 1)
        
        # If nothing else works, delete all digits -> 0, cost = L
        ans = min(ans, L)
        
        return ans