from collections import deque

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        visited = set()
        queue = deque()
        visited.add("")
        result = []
        queue.append("")
        while queue:
            current = queue.popleft()
            result.append(current)
            if len(current) < len(target):
                next1 = current + 'a'
                if next1 not in visited:
                    visited.add(next1)
                    queue.append(next1)
            if len(current) > 0:
                last_char = current[-1]
                if last_char != 'a':
                    next2 = current[:-1] + chr(ord(current[-1]) + 1)
                    if next2 not in visited:
                        visited.add(next2)
                        queue.append(next2)
        return result