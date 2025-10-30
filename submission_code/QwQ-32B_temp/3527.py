class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        if n == 0:
            return []
        
        # Compute the transitions array T
        T = [0] * n
        for i in range(n):
            next_i = (i + 1) % n
            T[i] = 1 if colors[i] != colors[next_i] else 0
        
        # Function to compute runs from the transitions array
        def compute_runs():
            runs = []
            if n == 0:
                return runs
            current_val = T[0]
            current_length = 1
            start = 0
            for i in range(1, n):
                if T[i] == current_val:
                    current_length += 1
                else:
                    runs.append((current_val, current_length, start, start + current_length - 1))
                    current_val = T[i]
                    current_length = 1
                    start = i
            runs.append((current_val, current_length, start, start + current_length - 1))
            return runs
        
        # Initial runs computation
        runs = compute_runs()
        first_run = runs[0] if runs else None
        last_run = runs[-1] if runs else None
        
        ans = []
        
        for q in queries:
            if q[0] == 2:
                # Update query: change colors[index_i] to color_i
                idx = q[1]
                new_color = q[2]
                old_color = colors[idx]
                colors[idx] = new_color
                
                # Update transitions at positions (idx-1) mod n and idx
                affected = [(idx - 1) % n, idx]
                for pos in affected:
                    prev = (pos - 1) % n
                    next_p = (pos + 1) % n
                    T[pos] = 1 if colors[prev] != colors[next_p] else 0
                
                # Recompute the runs after the update
                runs = compute_runs()
                first_run = runs[0] if runs else None
                last_run = runs[-1] if runs else None
            else:
                # Query of type 1: count groups of size size_i
                L = q[1] - 1  # L is the required length of transitions (size_i - 1)
                total = 0
                
                # Sum contributions from all runs of 1's
                for run in runs:
                    val, length, _, _ = run
                    if val == 1:
                        if length >= L:
                            total += (length - L + 1)
                
                # Check for circular contribution if first and last runs are both 1's
                if first_run and last_run:
                    first_val, first_len, _, _ = first_run
                    last_val, last_len, _, _ = last_run
                    if first_val == 1 and last_val == 1:
                        combined_length = first_len + last_len
                        if combined_length >= L:
                            total += (combined_length - L + 1)
                
                ans.append(total)
        
        return ans