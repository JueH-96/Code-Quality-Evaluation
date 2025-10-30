class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        original = [ord(c) - ord('a') for c in caption]
        INF = float('inf')
        
        pointers = []  # List to store pointers for each step
        
        # Initialize previous cost and pointers for the first character (i=0)
        prev_cost = [[INF] * 3 for _ in range(26)]
        prev_ptr = [[(None, None) for _ in range(3)] for _ in range(26)]
        
        for c in range(26):
            cost = abs(c - original[0])
            prev_cost[c][0] = cost  # run length 1 (index 0)
            prev_ptr[c][0] = (None, None)  # no previous
        
        # Append the initial pointers (deep copied)
        pointers.append([row.copy() for row in prev_ptr])
        
        for i in range(1, n):
            curr_cost = [[INF] * 3 for _ in range(26)]
            curr_ptr = [[(None, None) for _ in range(3)] for _ in range(26)]
            
            current_char = original[i]
            
            for prev_c in range(26):
                for prev_k in range(3):
                    if prev_cost[prev_c][prev_k] == INF:
                        continue
                    prev_total = prev_cost[prev_c][prev_k]
                    
                    for current_c in range(26):
                        delta = abs(current_c - current_char)
                        new_total = prev_total + delta
                        
                        if current_c == prev_c:
                            # Continuing the run
                            new_run_length = prev_k + 1
                            if new_run_length > 2:
                                new_run_length = 2  # cap at 3 (represented by index 2)
                            new_k = new_run_length
                            if new_k < 0:
                                continue
                            if new_total < curr_cost[current_c][new_k]:
                                curr_cost[current_c][new_k] = new_total
                                curr_ptr[current_c][new_k] = (prev_c, prev_k)
                            elif new_total == curr_cost[current_c][new_k]:
                                # Prefer the path with smaller previous_c if possible for lex order
                                # But since current_c is fixed, no change needed here
                                pass
                        else:
                            # Starting a new run, must have previous run valid (prev_k == 2)
                            if prev_k != 2:
                                continue
                            # New run length 1 (index 0)
                            new_k = 0
                            if new_total < curr_cost[current_c][new_k]:
                                curr_cost[current_c][new_k] = new_total
                                curr_ptr[current_c][new_k] = (prev_c, prev_k)
                            elif new_total == curr_cost[current_c][new_k]:
                                # Prefer the path with smaller previous_c if possible for lex order
                                pass
            
            # Append current pointers to the list (shallow copy is sufficient)
            pointers.append([row.copy() for row in curr_ptr])
            prev_cost = curr_cost
            prev_ptr = curr_ptr
        
        # Find the minimal cost among all final states (run length 3, index 2)
        min_cost = INF
        candidates = []
        for c in range(26):
            cost = prev_cost[c][2]
            if cost < min_cost:
                min_cost = cost
                candidates = [(c, 2)]
            elif cost == min_cost:
                candidates.append((c, 2))
        
        if min_cost == INF:
            return ""
        
        # Backtrack to find the lex smallest string among candidates
        min_str = None
        for (c, k) in candidates:
            path = []
            current_c = c
            current_k = k
            current_i = n - 1
            path.append(current_c)
            while current_i > 0:
                current_i -= 1
                # Get the previous step's pointers (pointers[current_i+1])
                prev_c, prev_k = pointers[current_i + 1][current_c][current_k]
                path.append(prev_c)
                current_c, current_k = prev_c, prev_k
            # Reverse to get the correct order and convert to string
            path.reverse()
            s = ''.join(chr(ord('a') + char) for char in path)
            if min_str is None or s < min_str:
                min_str = s
        
        return min_str if min_str else ""