class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        def is_good_caption(s):
            i = 0
            while i < len(s):
                count = 1
                while i + 1 < len(s) and s[i] == s[i + 1]:
                    count += 1
                    i += 1
                if count < 3:
                    return False
                i += 1
            return True
        
        def get_neighbors(c):
            neighbors = []
            if c != 'a':
                neighbors.append(chr(ord(c) - 1))
            if c != 'z':
                neighbors.append(chr(ord(c) + 1))
            neighbors.append(c)
            return sorted(neighbors)
        
        def try_all_combinations(pos, current, memo):
            if pos >= n:
                if is_good_caption(current):
                    return current
                return None
            
            key = (pos, current)
            if key in memo:
                return memo[key]
            
            best_result = None
            for neighbor in get_neighbors(caption[pos]):
                result = try_all_combinations(pos + 1, current + neighbor, memo)
                if result is not None:
                    if best_result is None or result < best_result:
                        best_result = result
            
            memo[key] = best_result
            return best_result
        
        # Try to find a solution
        result = try_all_combinations(0, "", {})
        return "" if result is None else result