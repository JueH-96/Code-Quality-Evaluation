class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        from itertools import combinations
        
        freq = Counter(s)
        n = len(s)
        
        # Convert to array for easier processing
        current = [0] * 26
        for char, count in freq.items():
            current[ord(char) - ord('a')] = count
        
        min_cost = float('inf')
        
        # Try all possible target configurations
        for num_distinct in range(1, 27):
            max_freq = n // num_distinct + 10 if num_distinct <= n else 10
            for common_freq in range(1, max_freq + 1):
                cost = self.minCostForConfig(current, num_distinct, common_freq)
                min_cost = min(min_cost, cost)
        
        return min_cost
    
    def minCostForConfig(self, current, num_distinct, common_freq):
        min_cost = float('inf')
        
        if num_distinct <= 15:  # For small numbers, try all combinations
            from itertools import combinations
            for chars in combinations(range(26), num_distinct):
                cost = self.calculateTransformCost(current, list(chars), common_freq)
                min_cost = min(min_cost, cost)
        else:
            # For large numbers, use greedy heuristic (pick most frequent chars)
            char_freqs = [(current[i], i) for i in range(26)]
            char_freqs.sort(reverse=True)
            chars = [char for _, char in char_freqs[:num_distinct]]
            cost = self.calculateTransformCost(current, chars, common_freq)
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def calculateTransformCost(self, current, target_chars, common_freq):
        current = current[:]
        target = [0] * 26
        for char in target_chars:
            target[char] = common_freq
        
        cost = 0
        
        # Process from left to right to handle change operation constraint
        for i in range(26):
            if current[i] > target[i]:
                excess = current[i] - target[i]
                
                # Try to change some to next character if beneficial
                if i < 25:
                    deficit_next = max(0, target[i+1] - current[i+1])
                    change_count = min(excess, deficit_next)
                    current[i+1] += change_count
                    cost += change_count
                    excess -= change_count
                
                # Delete remaining excess
                cost += excess
            
            elif current[i] < target[i]:
                deficit = target[i] - current[i]
                cost += deficit  # Insert needed characters
        
        return cost