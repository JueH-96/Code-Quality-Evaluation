class Solution:
    def maxOperations(self, s: str) -> int:
        s = list(s)
        operations = 0
        
        while True:
            moved = False
            # Find the leftmost movable '1'
            for i in range(len(s) - 1):
                if s[i] == '1' and s[i + 1] == '0':
                    # Find where this '1' should move to
                    j = i + 1
                    while j < len(s) and s[j] == '0':
                        j += 1
                    # Move s[i] to position j - 1
                    s[i] = '0'
                    s[j - 1] = '1'
                    operations += 1
                    moved = True
                    break
            
            if not moved:
                break
        
        return operations