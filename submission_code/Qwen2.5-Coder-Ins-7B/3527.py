class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        colors = colors + colors[:2]  # Extend colors to handle circular nature
        groups = []
        i = 0
        while i < n:
            if i == 0 or colors[i] != colors[i - 1]:
                j = i + 1
                while j < n + 2 and colors[j] != colors[j - 1]:
                    j += 1
                groups.append((i, j - i))
                i = j
            else:
                i += 1
        
        def count_alternating_groups(size):
            count = 0
            for start, length in groups:
                if length == size:
                    count += 1
                elif length > size:
                    if (length - size) % 2 == 0:
                        count += 1
            return count
        
        result = []
        for query in queries:
            if query[0] == 1:
                result.append(count_alternating_groups(query[1]))
            elif query[0] == 2:
                index, color = query[1], query[2]
                if colors[index + n] != color:
                    colors[index + n] = color
                    if index > 0 and colors[index + n] != colors[index + n - 1]:
                        if colors[index + n] == colors[index + n + 1]:
                            groups.append((index + n, 2))
                        else:
                            for i, (start, length) in enumerate(groups):
                                if start == index + n:
                                    groups[i] = (start, length + 1)
                                    break
                                elif start > index + n:
                                    groups.insert(i, (index + n, 2))
                                    break
                    else:
                        for i, (start, length) in enumerate(groups):
                            if start == index + n:
                                groups.pop(i)
                                break
                            elif start > index + n:
                                groups.insert(i, (index + n, 1))
                                break
        return result