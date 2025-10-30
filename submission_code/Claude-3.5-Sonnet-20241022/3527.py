class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def isAlternating(arr, start, size):
            n = len(arr)
            for i in range(size-1):
                curr = (start + i) % n
                next = (start + i + 1) % n
                if arr[curr] == arr[next]:
                    return False
            return True
            
        def countGroups(arr, size):
            n = len(arr)
            count = 0
            for start in range(n):
                if isAlternating(arr, start, size):
                    count += 1
            return count
        
        result = []
        colors = colors.copy()
        
        for query in queries:
            if query[0] == 1:
                size = query[1]
                result.append(countGroups(colors, size))
            else:
                index = query[1]
                new_color = query[2]
                colors[index] = new_color
                
        return result