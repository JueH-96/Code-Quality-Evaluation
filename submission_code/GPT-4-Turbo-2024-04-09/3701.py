class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        from collections import defaultdict
        import heapq
        
        if len(caption) < 3:
            return ""
        
        # This function will attempt to make all characters in the range [start, end] (inclusive)
        # the same character, and return the cost and resulting character
        def min_cost_to_uniform(start, end):
            freq = defaultdict(int)
            for i in range(start, end + 1):
                freq[caption[i]] += 1
            
            # Find the minimum cost to convert all to one character
            min_cost = float('inf')
            best_char = None
            
            # Consider changing all to each character from 'a' to 'z'
            for target_char in range(ord('a'), ord('z') + 1):
                target_char = chr(target_char)
                cost = 0
                for char, count in freq.items():
                    if char != target_char:
                        # Calculate cost to change all `char` to `target_char`
                        cost += count * (abs(ord(char) - ord(target_char)))
                
                if cost < min_cost or (cost == min_cost and target_char < best_char):
                    min_cost = cost
                    best_char = target_char
            
            return min_cost, best_char
        
        # We need to group characters in blocks of at least 3
        n = len(caption)
        i = 0
        operations = 0
        result = []
        
        while i < n:
            j = i
            while j + 1 < n and caption[j + 1] == caption[i]:
                j += 1
            
            length = j - i + 1
            if length >= 3:
                result.append(caption[i] * length)
            else:
                # Need to extend this group to at least 3
                if j + 1 < n and j - i + 1 < 3:
                    # Try to extend to the right or change the group
                    needed = 3 - (j - i + 1)
                    if j + needed < n:
                        cost, char = min_cost_to_uniform(i, j + needed)
                        if char is not None:
                            operations += cost
                            result.append(char * (j + needed - i + 1))
                            j += needed
                        else:
                            return ""
                    else:
                        return ""
                else:
                    return ""
            
            i = j + 1
        
        # Join the result and return
        return ''.join(result)