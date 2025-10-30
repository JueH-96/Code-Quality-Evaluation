class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        patterns = ["00", "25", "50", "75"]
        INF = float('inf')
        best = INF
        
        # Try to form each of the endings "00", "25", "50", "75"
        for pat in patterns:
            a, b = pat[0], pat[1]
            # find position i of b from right
            # moves1 = how many deletions needed after i to bring b to the last place
            i = -1
            moves1 = 0
            for idx in range(n - 1, -1, -1):
                if num[idx] == b:
                    i = idx
                    moves1 = (n - 1 - idx)
                    break
            if i == -1:
                continue
            # now find a to the left of i
            j = -1
            moves2 = 0
            for idx in range(i - 1, -1, -1):
                if num[idx] == a:
                    j = idx
                    moves2 = (i - 1 - idx)
                    break
            if j == -1:
                continue
            best = min(best, moves1 + moves2)
        
        # If we can keep exactly one '0' and delete all other digits,
        # that yields "0" which is divisible by 25.
        if '0' in num:
            best = min(best, n - 1)
        
        # If nothing else works, we can delete all digits to get 0
        best = min(best, n)
        
        return best