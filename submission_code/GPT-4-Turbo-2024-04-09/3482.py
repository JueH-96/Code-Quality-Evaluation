class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import defaultdict
        import heapq
        
        # Create a dictionary to store all positions of substrings in target
        word_positions = defaultdict(list)
        for i, word in enumerate(words):
            start = 0
            while True:
                start = target.find(word, start)
                if start == -1:
                    break
                word_positions[start].append((i, len(word)))  # Store index and length of word
                start += 1
        
        # Priority queue for minimum cost dynamic programming
        # (cost, position in target)
        heap = [(0, 0)]  # Start with cost 0 at position 0
        visited = [float('inf')] * (len(target) + 1)
        visited[0] = 0
        
        while heap:
            current_cost, pos = heapq.heappop(heap)
            
            # If we reached the end of the target string
            if pos == len(target):
                return current_cost
            
            # If this position has already been reached with a cheaper cost, skip it
            if current_cost > visited[pos]:
                continue
            
            # Explore all words that can start at this position
            for index, length in word_positions.get(pos, []):
                new_pos = pos + length
                new_cost = current_cost + costs[index]
                
                # Only consider this new position if it offers a cheaper way to get there
                if new_cost < visited[new_pos]:
                    visited[new_pos] = new_cost
                    heapq.heappush(heap, (new_cost, new_pos))
        
        # If we never reached the end of the target
        return -1