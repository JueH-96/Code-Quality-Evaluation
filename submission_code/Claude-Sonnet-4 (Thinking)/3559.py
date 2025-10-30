from collections import deque

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        visited = [False] * (n + 1)
        queue = deque([(0, 0)])  # (position, count)
        visited[0] = True
        
        while queue:
            pos, count = queue.popleft()
            
            if pos == n:
                return count
            
            for word in words:
                # Check all prefixes of this word that can start at pos
                max_len = min(len(word), n - pos)
                for i in range(1, max_len + 1):
                    if word[:i] == target[pos:pos + i]:
                        new_pos = pos + i
                        if not visited[new_pos]:
                            visited[new_pos] = True
                            queue.append((new_pos, count + 1))
                    else:
                        break  # No longer matching
        
        return -1