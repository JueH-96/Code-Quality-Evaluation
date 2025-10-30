class Solution:
    def smallestString(self, s: str) -> str:
        # Convert string to list for mutability
        arr = list(s)
        n = len(arr)

        # 1) Find the first character that is not 'a'
        i = 0
        while i < n and arr[i] == 'a':
            i += 1

        # 2) If all characters are 'a', we must decrement exactly one character:
        #    choose the last one, turning 'a' -> 'z'.
        if i == n:
            arr[-1] = 'z'
            return ''.join(arr)

        # 3) From the first non-'a', keep decrementing until we hit an 'a'
        j = i
        while j < n and arr[j] != 'a':
            # decrement character, with wrap 'a' -> 'z'
            if arr[j] == 'a':
                arr[j] = 'z'
            else:
                arr[j] = chr(ord(arr[j]) - 1)
            j += 1

        return ''.join(arr)