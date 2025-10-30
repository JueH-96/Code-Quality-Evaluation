class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        top = []
        for i in range(m):
            arr = [(j, board[i][j]) for j in range(n)]
            arr.sort(key=lambda x: x[1], reverse=True)
            top.append(arr[:3])
        
        ans = -10**18
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    for ci in top[i]:
                        for cj in top[j]:
                            for ck in top[k]:
                                cols = {ci[0], cj[0], ck[0]}
                                if len(cols) == 3:
                                    total = ci[1] + cj[1] + ck[1]
                                    if total > ans:
                                        ans = total
        return ans