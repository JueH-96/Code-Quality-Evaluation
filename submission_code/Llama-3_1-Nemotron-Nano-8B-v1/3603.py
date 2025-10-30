from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        for child_list in children:
            child_list.sort()
        
        base = 911382629
        mod = 10**18 + 3
        max_power = n
        power = [1] * (max_power + 1)
        for i in range(1, max_power + 1):
            power[i] = (power[i-1] * base) % mod
        
        post_order = []
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                post_order.append(node)
        
        h = [0] * n
        rh = [0] * n
        len_arr = [0] * n
        answer = [False] * n
        
        for node in post_order:
            kids = children[node]
            k = len(kids)
            suffix_sums = [0] * k
            sum_so_far = 0
            for i in reversed(range(k)):
                suffix_sums[i] = sum_so_far
                sum_so_far += len_arr[kids[i]]
            
            h_children = 0
            for i in range(k):
                h_children = (h_children * power[suffix_sums[i]] + h[kids[i]]) % mod
            
            reversed_kids = kids[::-1]
            k_rev = k
            suffix_sums_rev = [0] * k_rev
            sum_so_far_rev = 0
            for i in reversed(range(k_rev)):
                suffix_sums_rev[i] = sum_so_far_rev
                sum_so_far_rev += len_arr[reversed_kids[i]]
            
            rh_children_rev = 0
            for i in range(k_rev):
                rh_children_rev = (rh_children_rev * power[suffix_sums_rev[i]] + rh[reversed_kids[i]]) % mod
            
            current_char = ord(s[node])
            h[node] = (h_children * base + current_char) % mod
            rh[node] = (rh_children_rev * base + current_char) % mod
            len_arr[node] = k + 1
            answer[node] = (h[node] == rh[node])
        
        return answer