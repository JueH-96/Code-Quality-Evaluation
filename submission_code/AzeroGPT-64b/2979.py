class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort()
        offer_map = defaultdict(list)
        for i in range(n):
            offer_map[i].append(-1)
        
        for idx, (s, e, g) in enumerate(offers):
            offer_map[e + 1].append(idx)

        @cache
        def dfs(i):
            ans = 0
            for offer_idx in offer_map[i]:
                if offer_idx == -1:
                    ans = max(ans, dfs(i + 1))
                else:
                    s, e, g = offers[offer_idx]
                    ans = max(ans, g + dfs(e + 1))
            return ans

        return dfs(0)