class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        m = k.bit_length()
        nxt = [[(0) for _ in range(n)] for _ in range(m)]
        score = [[(0) for _ in range(n)] for _ in range(m)]
        for i in range(n):
            nxt[0][i] = receiver[i]
            score[0][i] = receiver[i]
        for i in range(1, m):
            for j in range(n):
                nxt[i][j] = nxt[i - 1][nxt[i - 1][j]]
                score[i][j] = score[i - 1][j] + score[i - 1][nxt[i - 1][j]]
        ans = 0
        for i in range(n):
            cur = i
            res = i
            for j in range(m):
                if (k >> j) & 1:
                    res += score[j][cur]
                    cur = nxt[j][cur]
            ans = max(ans, res)
        return ans