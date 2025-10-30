class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        if n == 0:
            return []
        
        # Precompute the difference array
        diff = [0] * n
        for i in range(n):
            j = (i + 1) % n
            if colors[i] != colors[j]:
                diff[i] = 1
        
        answers = []
        
        for q in queries:
            if q[0] == 1:
                s = q[1]
                count = 0
                if s == 1:
                    # Any single element is a valid group
                    count = n
                else:
                    for i in range(n):
                        total = 0
                        for j in range(s - 1):
                            pos = (i + j) % n
                            total += diff[pos]
                        if total == s - 1:
                            count += 1
                answers.append(count)
            else:
                # Update query
                index = q[1]
                new_color = q[2]
                if colors[index] == new_color:
                    continue  # No change needed
                colors[index] = new_color
                # Recompute the diff array
                for i in range(n):
                    j = (i + 1) % n
                    if colors[i] != colors[j]:
                        diff[i] = 1
                    else:
                        diff[i] = 0
        
        return answers