class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        n_char = 26
        mat = [[0] * n_char for _ in range(n_char)]
        
        for a in range(n_char):
            for b in range(n_char):
                if a == b:
                    mat[a][b] = 0
                else:
                    k_forward = (b - a) % n_char
                    cur = a
                    total_forward = 0
                    for _ in range(k_forward):
                        total_forward += nextCost[cur]
                        cur = (cur + 1) % n_char
                    
                    k_backward = n_char - k_forward
                    cur = a
                    total_backward = 0
                    for _ in range(k_backward):
                        total_backward += previousCost[cur]
                        cur = (cur - 1) % n_char
                    
                    mat[a][b] = min(total_forward, total_backward)
        
        total_cost = 0
        for i in range(len(s)):
            a = ord(s[i]) - ord('a')
            b = ord(t[i]) - ord('a')
            total_cost += mat[a][b]
        
        return total_cost