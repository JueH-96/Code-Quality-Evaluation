class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: list[int]) -> int:
        n = len(source)
        m = len(pattern)
        is_target = [False] * n
        for idx in targetIndices:
            is_target[idx] = True
        
        prev_min_counts = [float('inf')] * n
        
        for step in range(1, m + 1):
            curr_min_counts = [float('inf')] * n
            current_min = 0 if step == 1 else float('inf')
            
            for j in range(n):
                # Before processing j, current_min is the minimal up to j-1
                if source[j] == pattern[step - 1]:
                    if current_min != float('inf'):
                        curr_min_counts[j] = current_min + (1 if is_target[j] else 0)
                
                # Update current_min with prev_min_counts[j]
                if prev_min_counts[j] < float('inf'):
                    if current_min == float('inf'):
                        current_min = prev_min_counts[j]
                    else:
                        current_min = min(current_min, prev_min_counts[j])
            
            prev_min_counts = curr_min_counts
        
        # Find the minimal count among all valid ending positions for the last character
        min_count = float('inf')
        last_char = pattern[-1]
        for j in range(n):
            if source[j] == last_char and prev_min_counts[j] < min_count:
                min_count = prev_min_counts[j]
        
        return len(targetIndices) - min_count