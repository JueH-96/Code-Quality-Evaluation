from typing import List

def dijkstra(N: int, A: int, B: int, C: int, D: List[List[int]]) -> int:
    """
    Finds the shortest time to travel from city 1 to city N using a combination of
    company car and train transportation, with the constraint of only switching from
    company car to train, not vice versa.

    Parameters:
    - N: int - Number of cities.
    - A: int - Time multiplier for company car travel.
    - B: int - Time multiplier for train travel.
    - C: int - Time constant added to train travel.
    - D: List[List[int]] - Matrix representing travel distances between cities.

    Returns:
    - int: Minimum time to travel from city 1 to city N.
    """
    from heapq import heappop, heappush
    INF = 10**18

    car = [[D[i][j] * A for j in range(N)] for i in range(N)]
    train = [[D[i][j] * B + C if i < j else INF for j in range(N)] for i in range(N)]
    
    for k in range(N):
        for i in range(N):
            for j in range(N):
                car[i][j] = min(car[i][j], car[i][k] + car[k][j])
                if i < j:
                    train[i][j] = min(train[i][j], train[i][k] + train[k][j])
                    car[i][j] = min(car[i][j], train[i][k] + car[k][j])

    return min(car[0][N-1], train[0][N-1])

# Read input
N, A, B, C = map(int, input().split())
D = [list(map(int, input().split())) for _ in range(N)]

# Solve and print the result
print(dijkstra(N, A, B, C, D))