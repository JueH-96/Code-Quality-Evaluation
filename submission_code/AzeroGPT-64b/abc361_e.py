from typing import List, Tuple

def dfs(city: int, parent: int, distances: List[List[Tuple[int, int]]], costs: List[int]) -> Tuple[int, int]:
    min_cost_to_reach = costs[city]
    distance = 0
    for next_city, cost in distances[city]:
        if next_city == parent:
            continue
        next_cost, next_distance = dfs(next_city, city, distances, costs)
        min_cost_to_reach = min(min_cost_to_reach, next_cost)
        distance += next_distance + cost
    return min_cost_to_reach, distance + min(min_cost_to_reach, costs[city])*2

def main() -> None:
    n = int(input())
    costs = [1000000001] * n # Extreme high cost for unvisited cities.
    costs[0] = 0
    distances = [[] for _ in range(n)]
    for _ in range(n-1):
        a, b, c = map(int, input().split())
        a -= 1
        b -= 1
        distances[a].append((b, c))
        distances[b].append((a, c))
    _, distance = dfs(0, -1, distances, costs)
    print(distance + min(costs[1:])*2)

if __name__ == "__main__":
    main()