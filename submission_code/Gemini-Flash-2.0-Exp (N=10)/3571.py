class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        
        
        def find_max_path(start_index):
            
            
            def dfs(curr_index, path):
                max_len = len(path)
                
                for next_index in range(n):
                    if next_index != curr_index:
                        
                        if coordinates[next_index][0] > coordinates[curr_index][0] and coordinates[next_index][1] > coordinates[curr_index][1]:
                            max_len = max(max_len, dfs(next_index, path + [next_index]))
                return max_len
            
            return dfs(start_index, [start_index])
        
        
        return find_max_path(k)