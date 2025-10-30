from collections import defaultdict

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        prefix = [0] * (n + 1)
        for i, c in enumerate(colors):
            prefix[i + 1] = prefix[i] + (1 if i > 0 and colors[i - 1] != c else 0)
        
        def count_groups(size):
            return max(0, (prefix[-1] - prefix[size - 1]) // 2 + 1)
        
        answer = []
        changes = defaultdict(list)
        for i, q in reversed(list(enumerate(queries))):
            if q[0] == 2:
                changes[q[1]].append((i, q[2]))
        
        for i, c in enumerate(colors):
            for idx, val in changes[i]:
                queries[idx][1] = val
            colors[i] = c ^ sum((q[1] != colors[i]) for q in changes[i]) & 1
        
        for q in queries:
            if q[0] == 1:
                answer.append(count_groups(q[1]))
            else:
                colors[q[1]] = q[2]
                if q[1] > 0:
                    prefix[q[1] + 1] = prefix[q[1]] = prefix[q[1] - 1] + (1 if colors[q[1] - 1] != colors[q[1]] else 0)
                else:
                    prefix[1] = (1 if colors[1] != colors[0] else 0)
                if q[1] < n - 1:
                    prefix[q[1] + 2] = prefix[q[1] + 1] + (1 if colors[q[1] + 1] != colors[q[1] + 2] else 0)
        
        return answer[::-1]