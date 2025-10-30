class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        symmetric_count = 0
        
        for num in range(low, high + 1):
            s = str(num)
            # We only consider numbers with an even number of digits
            if len(s) % 2 == 0:
                half = len(s) // 2
                # Sum of first half and second half of the digits
                if sum(int(d) for d in s[:half]) == sum(int(d) for d in s[half:]):
                    symmetric_count += 1
                    
        return symmetric_count