from collections import deque

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        if not target:
            return []
        
        queue = deque()
        queue.append("")
        
        visited = set()
        visited.add("")
        
        parent = {}
        parent[""] = None
        
        found = False
        
        while queue:
            current = queue.popleft()
            
            # Generate next states
            # Key1: append 'a'
            next1 = current + 'a'
            # Key2: change last character if possible
            if current:
                last_char = current[-1]
                if last_char == 'z':
                    new_last = 'a'
                else:
                    new_last = chr(ord(last_char) + 1)
                next2 = current[:-1] + new_last
            else:
                next2 = None
            
            # Process next1
            if next1 not in visited:
                if next1 == target:
                    parent[next1] = current
                    found = True
                    visited.add(next1)  # Mark target as visited
                else:
                    visited.add(next1)
                    parent[next1] = current
                    queue.append(next1)
            
            # Process next2 if applicable
            if current and next2 is not None:
                if next2 not in visited:
                    if next2 == target:
                        parent[next2] = current
                        found = True
                        visited.add(next2)  # Mark target as visited
                    else:
                        visited.add(next2)
                        parent[next2] = current
                        queue.append(next2)
            
            if found:
                break
        
        if not found:
            return []
        
        # Reconstruct the path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = parent.get(current, None)
        
        # Reverse to get the order from start to end
        path.reverse()
        
        # Exclude the empty string
        result = [s for s in path if s != ""]
        
        return result