from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        def solve(path1, path2, path3):
            collected = 0
            visited = set()
            
            for r, c in path1:
                if (r, c) not in visited:
                    collected += fruits[r][c]
                    visited.add((r, c))
            
            for r, c in path2:
                if (r, c) not in visited:
                    collected += fruits[r][c]
                    visited.add((r, c))
            
            for r, c in path3:
                if (r, c) not in visited:
                    collected += fruits[r][c]
                    visited.add((r, c))
            
            return collected

        def generate_paths(n):
            paths1 = []
            paths2 = []
            paths3 = []

            def find_path1(r, c, current_path):
                if r == n - 1 and c == n - 1:
                    paths1.append(current_path + [(r,c)])
                    return
                
                if r + 1 < n and c + 1 < n:
                    find_path1(r + 1, c + 1, current_path + [(r,c)])
                if r + 1 < n:
                    find_path1(r + 1, c, current_path + [(r,c)])
                if c + 1 < n:
                    find_path1(r, c + 1, current_path + [(r,c)])

            def find_path2(r, c, current_path):
                if r == n - 1 and c == n - 1:
                    paths2.append(current_path + [(r,c)])
                    return
                
                if r + 1 < n and c - 1 >= 0:
                    find_path2(r + 1, c - 1, current_path + [(r,c)])
                if r + 1 < n:
                    find_path2(r + 1, c, current_path + [(r,c)])
                if r + 1 < n and c + 1 < n:
                    find_path2(r + 1, c + 1, current_path + [(r,c)])
            
            def find_path3(r, c, current_path):
                if r == n - 1 and c == n - 1:
                    paths3.append(current_path + [(r,c)])
                    return
                
                if r - 1 >= 0 and c + 1 < n:
                    find_path3(r - 1, c + 1, current_path + [(r,c)])
                if c + 1 < n:
                    find_path3(r, c + 1, current_path + [(r,c)])
                if r + 1 < n and c + 1 < n:
                    find_path3(r + 1, c + 1, current_path + [(r,c)])

            find_path1(0, 0, [])
            find_path2(0, n - 1, [])
            find_path3(n - 1, 0, [])
            
            return paths1, paths2, paths3

        paths1, paths2, paths3 = generate_paths(n)
        max_fruits = 0
        
        for path1 in paths1:
            for path2 in paths2:
                for path3 in paths3:
                    max_fruits = max(max_fruits, solve(path1, path2, path3))
        
        return max_fruits