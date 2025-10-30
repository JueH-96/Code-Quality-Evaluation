import collections

class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        def apply_operation1(current_s, i):
            s_list = list(current_s)
            for j in range(i + 1):
                s_list[j] = '1' if s_list[j] == '0' else '0'
            return "".join(s_list)
        def apply_operation2(current_s, i):
            s_list = list(current_s)
            for j in range(i, n):
                s_list[j] = '1' if s_list[j] == '0' else '0'
            return "".join(s_list)
        def is_all_zeros(current_s):
            return all(c == '0' for c in current_s)
        def is_all_ones(current_s):
            return all(c == '1' for c in current_s)
            
        min_cost_to_zeros = float('inf')
        min_cost_to_ones = float('inf')
        
        # BFS to reach all zeros
        initial_state = s
        queue = collections.deque([(initial_state, 0)])
        visited_costs_zeros = {initial_state: 0}
        
        while queue:
            current_string, current_cost = queue.popleft()
            if is_all_zeros(current_string):
                min_cost_to_zeros = min(min_cost_to_zeros, current_cost)
                continue # No need to explore further from a target state
                
            for i in range(n):
                next_string_op1 = apply_operation1(current_string, i)
                cost_op1 = i + 1
                new_cost_op1 = current_cost + cost_op1
                if next_string_op1 not in visited_costs_zeros or new_cost_op1 < visited_costs_zeros[next_string_op1]:
                    visited_costs_zeros[next_string_op1] = new_cost_op1
                    queue.append((next_string_op1, new_cost_op1))
                    
            for i in range(n):
                next_string_op2 = apply_operation2(current_string, i)
                cost_op2 = n - i
                new_cost_op2 = current_cost + cost_op2
                if next_string_op2 not in visited_costs_zeros or new_cost_op2 < visited_costs_zeros[next_string_op2]:
                    visited_costs_zeros[next_string_op2] = new_cost_op2
                    queue.append((next_string_op2, new_cost_op2))
                    
        # BFS to reach all ones
        queue = collections.deque([(initial_state, 0)])
        visited_costs_ones = {initial_state: 0}
        
        while queue:
            current_string, current_cost = queue.popleft()
            if is_all_ones(current_string):
                min_cost_to_ones = min(min_cost_to_ones, current_cost)
                continue # No need to explore further from a target state
                
            for i in range(n):
                next_string_op1 = apply_operation1(current_string, i)
                cost_op1 = i + 1
                new_cost_op1 = current_cost + cost_op1
                if next_string_op1 not in visited_costs_ones or new_cost_op1 < visited_costs_ones[next_string_op1]:
                    visited_costs_ones[next_string_op1] = new_cost_op1
                    queue.append((next_string_op1, new_cost_op1))
                    
            for i in range(n):
                next_string_op2 = apply_operation2(current_string, i)
                cost_op2 = n - i
                new_cost_op2 = current_cost + cost_op2
                if next_string_op2 not in visited_costs_ones or new_cost_op2 < visited_costs_ones[next_string_op2]:
                    visited_costs_ones[next_string_op2] = new_cost_op2
                    queue.append((next_string_op2, new_cost_op2))
                    
        return min(min_cost_to_zeros, min_cost_to_ones)