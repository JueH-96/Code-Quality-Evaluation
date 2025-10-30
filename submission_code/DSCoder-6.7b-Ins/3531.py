class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        import heapq
        h = []
        for i in range(len(damage)):
            if damage[i] < health[i]:
                heapq.heappush(h, (damage[i]/health[i], damage[i], health[i]))
            else:
                heapq.heappush(h, (0, damage[i], health[i]))
        total_damage = 0
        while h and h[0][2] <= power:
            ratio, dmg, hp = heapq.heappop(h)
            total_damage += dmg
            if h:
                ratio2, dmg2, hp2 = heapq.heappop(h)
                total_damage += dmg2
                if hp2 - (power - hp) > 0:
                    heapq.heappush(h, (dmg2/(hp2 - (power - hp)), dmg2, hp2 - (power - hp)))
        return total_damage