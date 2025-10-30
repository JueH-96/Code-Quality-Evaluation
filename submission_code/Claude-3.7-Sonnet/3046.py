class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # Case 1: The number is already divisible by 25
        if n >= 2 and (num[-2:] in ["00", "25", "50", "75"]):
            return 0
        elif n == 1 and num == "0":
            return 0
        
        # Case 2: Make the number 0
        min_deletions = n
        if "0" in num:
            min_deletions = n - 1
        
        # Case 3: Delete digits to make the number end with a valid pattern
        valid_patterns = ["00", "25", "50", "75"]
        
        for pattern in valid_patterns:
            for j in range(n - 1, -1, -1):
                if num[j] == pattern[1]:
                    for i in range(j - 1, -1, -1):
                        if num[i] == pattern[0]:
                            # Number of digits to delete = All digits after j + All digits between i and j
                            deletions = (n - j - 1) + (j - i - 1)
                            min_deletions = min(min_deletions, deletions)
                            break
        
        return min_deletions