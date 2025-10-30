class Solution:
    def minimumLength(self, s: str) -> int:
        stack = []
        count = {}
        
        for char in s:
            if char in count:
                count[char] += 1
            else:
                count[char] = 1
            stack.append(char)
        
        while True:
            found = False
            for i in range(len(stack)):
                char = stack[i]
                if count[char] > 1:
                    left = i - 1
                    right = i + 1
                    
                    while left >= 0 and stack[left] != char:
                        left -= 1
                    while right < len(stack) and stack[right] != char:
                        right += 1
                    
                    if left >= 0 and right < len(stack):
                        found = True
                        count[char] -= 2
                        stack.pop(right)
                        stack.pop(left)
                        break
            
            if not found:
                break
        
        return len(stack)