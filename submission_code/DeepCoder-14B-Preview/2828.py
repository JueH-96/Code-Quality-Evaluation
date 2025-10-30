class Solution:
    def smallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        min_str = None
        
        # Find the leftmost character that can be decreased
        left = 0
        while left < n and s_list[left] == 'a':
            left += 1
        
        # If all characters are 'a's, shift the entire string
        if left == n:
            return ''.join(['z' if c == 'a' else chr(ord(c) - 1) for c in s_list])
        
        # Find the rightmost character from left that can be decreased
        right = left
        while right < n and s_list[right] != 'a':
            right += 1
        right -= 1
        
        # Create the new string by shifting left to right
        new_str = s_list.copy()
        for k in range(left, right + 1):
            if new_str[k] == 'a':
                new_str[k] = 'z'
            else:
                new_str[k] = chr(ord(new_str[k]) - 1)
        current_str = ''.join(new_str)
        min_str = current_str
        
        # Also consider shifting just the left character
        new_str = s_list.copy()
        new_str[left] = 'z' if new_str[left] == 'a' else chr(ord(new_str[left]) - 1)
        current_str = ''.join(new_str)
        if current_str < min_str:
            min_str = current_str
        
        return min_str