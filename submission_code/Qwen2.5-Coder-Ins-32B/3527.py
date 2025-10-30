from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        count = [0] * (n + 1)
        
        def update_count(index):
            nonlocal count
            left = index - 1
            right = (index + 1) % n
            
            # Remove the contribution of the current tile from the count
            if left >= 0 and colors[left] != colors[index]:
                length = 2
                l = left - 1
                r = (index + 1) % n
                while l >= 0 and colors[l] != colors[left]:
                    l -= 1
                    length += 1
                while r != index and colors[r] != colors[index]:
                    r = (r + 1) % n
                    length += 1
                count[length] -= 1
            
            # Add the contribution of the current tile to the count
            if left >= 0 and colors[left] != colors[index]:
                length = 2
                l = left - 1
                r = (index + 1) % n
                while l >= 0 and colors[l] != colors[left]:
                    l -= 1
                    length += 1
                while r != index and colors[r] != colors[index]:
                    r = (r + 1) % n
                    length += 1
                count[length] += 1
        
        # Initial count of alternating groups
        for i in range(n):
            if colors[i] != colors[(i + 1) % n]:
                length = 2
                l = i - 1
                r = (i + 2) % n
                while l >= 0 and colors[l] != colors[i]:
                    l -= 1
                    length += 1
                while r != (i + 1) % n and colors[r] != colors[(i + 1) % n]:
                    r = (r + 1) % n
                    length += 1
                count[length] += 1
        
        result = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                result.append(count[size])
            elif query[0] == 2:
                index = query[1]
                new_color = query[2]
                if colors[index] != new_color:
                    colors[index] = new_color
                    update_count(index)
                    update_count((index - 1) % n)
                    update_count((index + 1) % n)
        
        return result