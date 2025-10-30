class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        def get_gaps(n, sick):
            gaps = []
            last_sick = -1
            for s in sick:
                if s - last_sick > 1:
                    gaps.append((last_sick + 1, s - 1))
                last_sick = s
            if n - 1 - last_sick > 0:
                gaps.append((last_sick + 1, n - 1))
            return gaps

        def count_sequences_in_gap(start, end, initial_sick):
            if start > end:
                return 1
            
            memo = {}
            
            def solve(current_sick_mask, infected_children):
                state = (current_sick_mask)
                if state in memo:
                    return memo[state]
                
                non_sick_children = []
                for i in range(start, end + 1):
                    if not (current_sick_mask & (1 << (i - start))):
                        non_sick_children.append(i)
                        
                if not non_sick_children:
                    return 1

                infectable_children = []
                sick_positions = []
                for i in range(start, end + 1):
                    if (current_sick_mask & (1 << (i - start))):
                        sick_positions.append(i)
                sick_positions.extend(initial_sick)
                sick_positions = set(sick_positions)

                for child_pos in non_sick_children:
                    if (child_pos - 1 in sick_positions) or (child_pos + 1 in sick_positions):
                        infectable_children.append(child_pos)
                
                if not infectable_children:
                    return 0
                    
                count = 0
                unique_infectable = sorted(list(set(infectable_children)))
                for infect_child in unique_infectable:
                    if infect_child >= start and infect_child <= end and not (current_sick_mask & (1 << (infect_child - start))):
                        next_sick_mask = current_sick_mask | (1 << (infect_child - start))
                        count = (count + solve(next_sick_mask, infected_children + [infect_child])) % MOD
                        
                memo[state] = count
                return count

            length = end - start + 1
            initial_mask = 0
            return solve(initial_mask, [])

        gaps = get_gaps(n, sick)
        total_sequences = 1
        for start, end in gaps:
            is_left_sick = (start == 0) or (start - 1 in sick)
            is_right_sick = (end == n - 1) or (end + 1 in sick)
            
            sequences_for_gap = count_sequences_in_gap(start, end, sick)
            total_sequences = (total_sequences * sequences_for_gap) % MOD

        return total_sequences