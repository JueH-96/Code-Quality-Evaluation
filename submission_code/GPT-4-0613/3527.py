from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        colors.append(colors[0])
        n = len(colors)
        cnt = [0] * n
        group = [0] * n
        j = 0
        for i in range(1, n):
            if colors[i] != colors[i - 1]:
                j += 1
            group[i] = j
            cnt[j] += 1
        cnt = cnt[:j + 1]
        prefix = [0] * (j + 2)
        for i in range(1, j + 2):
            prefix[i] = prefix[i - 1] + cnt[i - 1]
        res = []
        for query in queries:
            if query[0] == 1:
                if query[1] > len(cnt):
                    res.append(0)
                else:
                    res.append(cnt[query[1] - 1])
            else:
                if colors[query[1]] == query[2]:
                    continue
                colors[query[1]] = query[2]
                left = (query[1] - 1 + n - 1) % (n - 1)
                right = (query[1] + 1) % (n - 1)
                if colors[left] == colors[query[1]] and colors[right] == colors[query[1]]:
                    if group[left] != group[right]:
                        cnt[cnt[left]] -= 1
                        cnt[cnt[right]] -= 1
                        cnt[cnt[left] + cnt[right] + 1] += 1
                elif colors[left] != colors[query[1]] and colors[right] != colors[query[1]]:
                    if group[left] == group[right]:
                        cnt[cnt[left]] -= 1
                        cnt[cnt[left] // 2] += 2
                else:
                    if group[left] == group[right]:
                        cnt[cnt[left]] -= 1
                        cnt[cnt[left] // 2 + 1] += 1
                        cnt[cnt[left] // 2] += 1
                    else:
                        cnt[cnt[left]] -= 1
                        cnt[cnt[right]] -= 1
                        cnt[cnt[left] + cnt[right] - 1] += 1
        return res