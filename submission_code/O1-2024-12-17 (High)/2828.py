class Solution:
    def smallestString(self, s: str) -> str:
        arr = list(s)
        n = len(arr)
        
        # Find the first position where character is not 'a'
        i = 0
        while i < n and arr[i] == 'a':
            i += 1
        
        # If all are 'a', we must change exactly one character (pick the last one)
        if i == n:
            arr[-1] = 'z'
            return ''.join(arr)
        
        # From this position, decrease characters until we reach an 'a' or the end
        while i < n and arr[i] != 'a':
            # Decrease the character by 1 ('b'->'a', 'c'->'b', …, 'a'->'z')
            # but we won't see an 'a' in this loop because we stop if we do
            arr[i] = chr((ord(arr[i]) - ord('a') - 1) % 26 + ord('a'))
            i += 1
        
        return ''.join(arr)