class Solution:
    def validSequence(self, word1: str, word2: str) -> list:
        n, m = len(word1), len(word2)
        p1, p2 = [0]*26, [0]*26
        nxt = [p1[:]]
        for i in range(n-1, -1, -1):
            p1[ord(word1[i])-97] = i
            nxt.append(p1[:])
        nxt.reverse()
        p1 = [0]*26
        res = []
        for i in range(m):
            if p1[ord(word2[i])-97] < n and (i == m-1 or nxt[p1[ord(word2[i])-97]+1][ord(word2[i+1])-97] < n):
                res.append(p1[ord(word2[i])-97])
                p1[ord(word2[i])-97] = nxt[p1[ord(word2[i])-97]+1][ord(word2[i])-97] + 1
            else:
                ok = False
                for j in range(26):
                    if p1[j] < n and (i == m-1 or nxt[p1[j]+1][ord(word2[i+1])-97] < n):
                        res.append(p1[j])
                        p1[j] = nxt[p1[j]+1][j] + 1
                        ok = True
                        break
                if not ok:
                    return []
        return res