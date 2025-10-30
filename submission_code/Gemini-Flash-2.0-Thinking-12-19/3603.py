from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children_list = [[] for _ in range(n)]
        for i in range(1, n):
            children_list[parent[i]].append(i)
        
        answer = [False] * n
        
        for start_node in range(n):
            dfs_str = ""
            
            def dfs(x):
                nonlocal dfs_str
                for child in children_list[x]:
                    dfs(child)
                dfs_str += s[x]
                
            dfs(start_node)
            
            def is_palindrome(text):
                l = len(text)
                if l <= 1:
                    return True
                for i in range(l // 2):
                    if text[i] != text[l - 1 - i]:
                        return False
                return True
                
            if is_palindrome(dfs_str):
                answer[start_node] = True
            else:
                answer[start_node] = False
                
        return answer