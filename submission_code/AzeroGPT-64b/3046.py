class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        ends_with_00 = 0
        ends_with_25 = 0
        ends_with_50 = 0
        
        for i in range(n-1, -1, -1):
            if num[i] == '0':
                ends_with_00 += ends_with_00_from_0
                ends_with_00_from_0 += 1
                ends_with_25 += int(num[i-1]) == 2 if i > 0 else 0
            elif num[i] == '5':
                ends_with_25 += int(num[i-1]) == 2 if i > 0 else 0
                ends_with_50 += int(num[i-1]) == 5 if i > 0 else 0
            elif num[i] == '2':
                ends_with_25 += int(num[i-1]) in ('0', '5') if i > 0 else 0
            elif num[i] == '7':
                ends_with_25 += int(num[i-1]) in ('2', '7') if i > 0 else 0
            ends_with_00_from_0 = 0 if num[i] != '0' else (n-i-1)
        
        if n in (ends_with_00, ends_with_25, ends_with_50):
            return n - (ends_with_00 or ends_with_25 or ends_with_50)
        
        return n - (ends_with_00 or ends_with_25 or ends_with_50) + 1