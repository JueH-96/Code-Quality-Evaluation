from collections import deque

class Solution:
    def minimizedStringLength(self, s: str) -> int:
        visited = set()
        queue = deque()
        queue.append(s)
        visited.add(s)
        min_length = len(s)
        
        while queue:
            current = queue.popleft()
            current_len = len(current)
            if current_len < min_length:
                min_length = current_len
                if min_length == 0:
                    break  # Can't get shorter than 0
            for i in range(len(current)):
                c = current[i]
                # Find the closest left occurrence
                left = -1
                for j in range(i - 1, -1, -1):
                    if current[j] == c:
                        left = j
                        break
                # Find the closest right occurrence
                right = -1
                for j in range(i + 1, len(current)):
                    if current[j] == c:
                        right = j
                        break
                # Generate the new string based on left and right found
                if left == -1 and right == -1:
                    continue  # No deletions possible for this i
                elif left == -1:
                    new_s = current[:right] + current[right+1:]
                elif right == -1:
                    new_s = current[:left] + current[left+1:]
                else:
                    new_s = current[:left] + current[left+1:right] + current[right+1:]
                if new_s not in visited:
                    visited.add(new_s)
                    queue.append(new_s)
        return min_length