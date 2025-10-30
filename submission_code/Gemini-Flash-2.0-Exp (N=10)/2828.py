class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        arr = list(s)
        start = -1
        for i in range(n):
            if arr[i] != 'a':
                start = i
                break
        if start == -1:
            arr[-1] = 'z'
            return "".join(arr)
        
        end = start
        while end < n and arr[end] != 'a':
            end += 1
        
        for i in range(start, end):
            arr[i] = chr(ord(arr[i]) - 1)
        
        return "".join(arr)