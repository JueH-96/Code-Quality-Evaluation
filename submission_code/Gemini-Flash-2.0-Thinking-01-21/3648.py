class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        memo = {}
        
        def get_fruits_value(r1, c1, r2, c2, r3, c3):
            rooms = set()
            if 0 <= r1 < n and 0 <= c1 < n:
                rooms.add((r1, c1))
            if 0 <= r2 < n and 0 <= c2 < n:
                rooms.add((r2, c2))
            if 0 <= r3 < n and 0 <= c3 < n:
                rooms.add((r3, c3))
            total_fruits = 0
            for r, c in rooms:
                total_fruits += fruits[r][c]
            return total_fruits

        def solve(step, c1, c2):
            if step == n:
                return 0
            if (step, c1, c2) in memo:
                return memo[(step, c1, c2)]
            
            r1, r2, r3 = step, step, n - 1
            c3 = step
            
            current_fruits = get_fruits_value(r1, c1, r2, c2, r3, c3)
            max_prev_fruits = 0
            
            possible_prev_c1s = []
            if c1 - 1 >= 0:
                possible_prev_c1s.append(c1 - 1)
            possible_prev_c1s.append(c1)
            
            possible_prev_c2s = []
            if c2 + 1 < n:
                possible_prev_c2s.append(c2 + 1)
            possible_prev_c2s.append(c2)
            if c2 - 1 >= 0:
                possible_prev_c2s.append(c2 - 1)
                
            max_val = 0
            
            for prev_c1 in possible_prev_c1s:
                if 0 <= prev_c1 < n:
                    for prev_c2 in possible_prev_c2s:
                        if 0 <= prev_c2 < n:
                            val = solve(step + 1, get_next_c1(prev_c1, c1), get_next_c2(prev_c2, c2))
                            max_val = max(max_val, val)
                            
            result = current_fruits + max_val
            memo[(step, c1, c2)] = result
            return result

        def get_next_c1(prev_c1, c1):
            return c1
            
        def get_next_c2(prev_c2, c2):
            return c2

        dp = {}
        
        def get_max_fruits_dp(step, c1, c2):
            if step == n:
                return 0
            if (step, c1, c2) in dp:
                return dp[(step, c1, c2)]
            
            r1, r2, r3 = step, step, n - 1
            c3 = step
            
            current_fruits = get_fruits_value(r1, c1, r2, c2, r3, c3)
            max_prev_fruits = 0
            
            max_val = 0
            
            for pc1 in [c1-1, c1]:
                if 0 <= pc1 < n:
                    for pc2 in [c2+1, c2, c2-1]:
                        if 0 <= pc2 < n:
                            prev_val = get_max_fruits_dp(step + 1, pc1, pc2)
                            max_val = max(max_val, prev_val)

            result = current_fruits + max_val
            dp[(step, c1, c2)] = result
            return result

        max_total_fruits = 0
        dp_table = {}

        dp_table[0] = {}
        for c1 in range(n):
            for c2 in range(n):
                dp_table[0][(c1, c2)] = -float('inf')
        dp_table[0][(0, n-1)] = get_fruits_value(0, 0, 0, n-1, n-1, 0)

        for step in range(1, n):
            dp_table[step] = {}
            for c1 in range(n):
                for c2 in range(n):
                    dp_table[step][(c1, c2)] = -float('inf')
                    for pc1 in [c1 - 1, c1]:
                        if 0 <= pc1 < n:
                            for pc2 in [c2 + 1, c2, c2 - 1]:
                                if 0 <= pc2 < n:
                                    if (pc1, pc2) in dp_table[step-1]:
                                        dp_table[step][(c1, c2)] = max(dp_table[step][(c1, c2)], dp_table[step-1][(pc1, pc2)])
                    if dp_table[step][(c1, c2)] != -float('inf'):
                        dp_table[step][(c1, c2)] += get_fruits_value(step, c1, step, c2, n-1, step)

        result = 0
        for c1 in range(n):
            for c2 in range(n):
                result = max(result, dp_table[n-1].get((c1, c2), 0))

        return result