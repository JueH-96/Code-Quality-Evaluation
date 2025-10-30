class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_r = 35
        
        nxt_table = [receiver.copy()]
        sums_table = [[i + receiver[i] for i in range(n)]]
        
        for r in range(1, max_r):
            nxt_prev = nxt_table[-1]
            sums_prev = sums_table[-1]
            nxt_cur = [0] * n
            sums_cur = [0] * n
            for i in range(n):
                mid = nxt_prev[i]
                nxt_cur[i] = nxt_prev[mid]
                sums_cur[i] = sums_prev[i] + sums_prev[mid] - mid
            nxt_table.append(nxt_cur)
            sums_table.append(sums_cur)
        
        ans = 0
        for j in range(n):
            total = j
            cur = j
            for r in range(max_r):
                if (k >> r) & 1:
                    total += sums_table[r][cur] - cur
                    cur = nxt_table[r][cur]
            if total > ans:
                ans = total
        return ans