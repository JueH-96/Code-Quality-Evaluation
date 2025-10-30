class Solution:
    def minimumLength(self, s: str) -> int:
        stack = []
        
        for char in s:
            if stack and stack[-1][0] == char:
                stack[-1][1] += 1
            else:
                stack.append([char, 1])
        
        while len(stack) > 1:
            if stack[-1][0] == stack[-2][0]:
                left_count = stack.pop()[1]
                right_count = stack.pop()[1]
                if left_count == 1 and right_count == 1:
                    continue
                elif left_count > right_count:
                    stack.append([stack[-1][0], left_count - right_count])
                elif right_count > left_count:
                    stack.append([stack[-1][0], right_count - left_count])
            else:
                break
        
        return sum(count for char, count in stack)