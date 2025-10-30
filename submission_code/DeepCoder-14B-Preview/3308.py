class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        current = s
        while True:
            indices = []
            for c in 'abcdefghijklmnopqrstuvwxyz':
                idx = current.find(c)
                if idx != -1:
                    indices.append(idx)
            if not indices:
                break
            # Remove duplicates and sort indices in reverse order
            indices = sorted(list(set(indices)), reverse=True)
            # Build the new current string by removing each index
            new_current = current
            for idx in indices:
                new_current = new_current[:idx] + new_current[idx+1:]
            # Check if the new string is empty
            if len(new_current) == 0:
                return current
            current = new_current
        return current