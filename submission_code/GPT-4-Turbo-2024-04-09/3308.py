class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import deque
        
        def remove_first_occurrences(s, alphabet):
            s_deque = deque(s)
            new_s = []
            used = set()
            
            while s_deque:
                char = s_deque.popleft()
                if char in alphabet and char not in used:
                    used.add(char)
                else:
                    new_s.append(char)
            
            return ''.join(new_s)
        
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        previous_s = s
        
        while True:
            s = remove_first_occurrences(s, alphabet)
            if not s:
                return previous_s
            previous_s = s