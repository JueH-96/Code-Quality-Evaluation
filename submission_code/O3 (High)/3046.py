class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # Worst-case : delete every digit -> the string becomes "0"
        ans = n
        
        # All possible pairs that make a number end with a multiple of 25
        targets = ("00", "25", "50", "75")
        
        for t in targets:
            second = t[1]
            # look for the unit digit (second) from the rightmost side
            for j in range(n - 1, -1, -1):
                if num[j] != second:
                    continue
                first = t[0]
                # look for the tens digit (first) to the left of j
                for i in range(j - 1, -1, -1):
                    if num[i] == first:
                        # number of deletions = everything after j  +  everything between i and j
                        # which simplifies to n - i - 2
                        ans = min(ans, n - i - 2)
                        break   # largest i gives fewest deletions for this j
        
        # Option: keep a single '0' (resulting number is "0")
        if '0' in num:
            ans = min(ans, n - 1)
        
        return ans