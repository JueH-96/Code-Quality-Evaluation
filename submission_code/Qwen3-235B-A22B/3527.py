class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        if n == 0:
            return []
        diff = [0] * n
        for i in range(n):
            next_i = (i + 1) % n
            diff[i] = 1 if colors[i] != colors[next_i] else 0
        
        res = []
        
        for q in queries:
            if q[0] == 2:
                # Type 2: update query
                index = q[1]
                new_color = q[2]
                colors[index] = new_color
                # Update diff[index-1] and diff[index]
                for pos in [(index - 1 + n) % n, index]:
                    next_pos = (pos + 1) % n
                    a = colors[pos]
                    b = colors[next_pos]
                    diff[pos] = 1 if a != b else 0
            else:
                # Type 1: count query
                k = q[1]
                s = k - 1
                count = 0
                if s < 1 or s > n:
                    # s must be at least 1 (k >=2, but problem says type 1 queries have size_i >=3)
                    # but per problem statement, queries of type 1 have k >=3, so s >=2
                    # handle invalid cases
                    res.append(0)
                    continue
                current_sum = 0
                # Initialize the first window
                for i in range(s):
                    current_sum += diff[i]
                if current_sum == s:
                    count += 1
                # Slide the window
                for start in range(1, n):
                    # Remove the element leaving the window
                    out_pos = (start - 1 + n) % n  # (start-1) mod n
                    current_sum -= diff[out_pos]
                    # Add the new element entering the window
                    in_pos = (start + s - 1) % n
                    current_sum += diff[in_pos]
                    if current_sum == s:
                        count += 1
                res.append(count)
        
        return res