class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total = 0
        for sc, tc in zip(s, t):
            current = ord(sc) - ord('a')
            target = ord(tc) - ord('a')
            if current == target:
                continue
            delta = (target - current) % 26
            steps_forward = delta
            steps_backward = 26 - delta
            
            # Calculate forward cost
            cost_forward = 0
            current_f = current
            for _ in range(steps_forward):
                cost_forward += nextCost[current_f]
                current_f = (current_f + 1) % 26
            
            # Calculate backward cost
            cost_backward = 0
            current_b = current
            for _ in range(steps_backward):
                cost_backward += previousCost[current_b]
                current_b = (current_b - 1) % 26
            
            total += min(cost_forward, cost_backward)
        
        return total