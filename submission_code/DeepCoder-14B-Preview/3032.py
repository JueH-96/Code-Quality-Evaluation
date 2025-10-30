class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_sum = 0
        
        for x in range(n):
            current = x
            path = []
            pos = {}
            
            while current not in pos:
                path.append(current)
                pos[current] = len(path) - 1
                current = receiver[current]
            
            if current in pos:
                cycle_start = pos[current]
                tail = path[:cycle_start]
                cycle = path[cycle_start:]
            else:
                # This case should not occur as per problem constraints
                tail = path
                cycle = []
            
            m = len(tail)
            c = len(cycle)
            sum_tail = sum(tail)
            sum_cycle = sum(cycle)
            
            # Compute prefix sums for the cycle
            prefix = [0]
            s = 0
            for node in cycle:
                s += node
                prefix.append(s)
            
            total_terms = k + 1
            
            if total_terms <= m:
                sum_x = sum(path[:total_terms])
            else:
                sum_tail_part = sum_tail
                steps_after_tail = total_terms - m
                if c == 0:
                    # This case should not occur as per problem constraints
                    sum_x = sum_tail_part
                else:
                    full_cycles = steps_after_tail // c
                    remaining = steps_after_tail % c
                    sum_cycle_part = full_cycles * sum_cycle
                    sum_remaining = prefix[remaining]
                    sum_x = sum_tail_part + sum_cycle_part + sum_remaining
            
            if sum_x > max_sum:
                max_sum = sum_x
        
        return max_sum