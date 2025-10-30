import heapq
from collections import defaultdict

class Solution:
    def minimumCost(self, target: str, words: list, costs: list) -> int:
        len_target = len(target)
        if len_target == 0:
            return 0
        
        # Precompute transitions: for each position i, a dictionary of possible lengths and minimal costs
        transitions = defaultdict(dict)  # transitions[i] is a dict mapping length to minimal cost
        
        for j in range(len(words)):
            word = words[j]
            len_w = len(word)
            if len_w == 0:
                continue  # skip empty words as per problem constraints
            cost = costs[j]
            
            start = 0
            while True:
                pos = target.find(word, start)
                if pos == -1:
                    break
                # Update the minimal cost for this (pos, len_w)
                current_min = transitions[pos].get(len_w, float('inf'))
                if cost < current_min:
                    transitions[pos][len_w] = cost
                start = pos + 1  # move past this occurrence to find next possible one
        
        # Initialize DP array and priority queue
        dp = [float('inf')] * (len_target + 1)
        dp[0] = 0
        heap = []
        heapq.heappush(heap, (0, 0))
        
        while heap:
            current_cost, current_i = heapq.heappop(heap)
            
            if current_i == len_target:
                return current_cost
            
            if current_cost > dp[current_i]:
                continue  # Skip if this is not the minimal cost to reach current_i
            
            # Process all possible transitions from current_i
            if current_i in transitions:
                for l in transitions[current_i]:
                    cost = transitions[current_i][l]
                    next_i = current_i + l
                    if next_i > len_target:
                        continue
                    new_cost = current_cost + cost
                    if new_cost < dp[next_i]:
                        dp[next_i] = new_cost
                        heapq.heappush(heap, (new_cost, next_i))
        
        # If we reach here, it's impossible to form the target
        return -1 if dp[len_target] == float('inf') else dp[len_target]