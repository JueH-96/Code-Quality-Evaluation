class Solution:
    def getSmallestString(self, s: str) -> str:
        even, odd = [], []
        for i, c in enumerate(s):
            if int(c) % 2 == 0:
                even.append((i, c))
            else:
                odd.append((i, c))
        
        even.sort(key=lambda x: x[1])
        odd.sort(key=lambda x: x[1])
        
        swapped = False
        i, j = 0, 0
        while i < len(even) and j < len(odd):
            if even[i][0] + 1 == odd[j][0]:
                swapped = True
                even[i], odd[j] = odd[j], even[i]
                i += 1
                j += 1
            elif even[i][0] < odd[j][0]:
                i += 1
            else:
                j += 1
        
        result = [''] * len(s)
        for i, c in even + odd:
            result[i] = c
        
        return ''.join(result) if swapped else s