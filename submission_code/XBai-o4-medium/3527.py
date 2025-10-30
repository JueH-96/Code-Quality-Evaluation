class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        # Initialize the diff array
        diff = [0] * n
        for i in range(n):
            j = (i + 1) % n
            if colors[i] != colors[j]:
                diff[i] = 1
            else:
                diff[i] = 0
        
        res = []
        for q in queries:
            if q[0] == 1:
                s = q[1]
                cnt = 0
                for i in range(n):
                    valid = True
                    for k in range(s - 1):
                        pos = (i + k) % n
                        if diff[pos] == 0:
                            valid = False
                            break
                    if valid:
                        cnt += 1
                res.append(cnt)
            else:
                index, color = q[1], q[2]
                if colors[index] == color:
                    continue  # No change needed
                colors[index] = color
                # Update the diff array for the affected positions
                for i in range(3):
                    pos = (index + n - 1 - i) % n
                    next_pos = (pos + 1) % n
                    if colors[pos] != colors[next_pos]:
                        diff[pos] = 1
                    else:
                        diff[pos] = 0
        return res