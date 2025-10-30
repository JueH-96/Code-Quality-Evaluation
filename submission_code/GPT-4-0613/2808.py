class Solution:
    def paintWalls(self, cost, time):
        n = len(cost)
        walls = sorted([(cost[i], time[i]) for i in range(n)], key=lambda x: (-x[1], x[0]))
        total_time = sum(time)
        total_cost = 0
        for c, t in walls:
            if total_time > t:
                total_time -= t
            else:
                total_cost += c
                total_time -= 1
        return total_cost