from collections import deque

def solve():
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    tasks = []
    for _ in range(Q):
        T, G = map(int, input().split())
        tasks.append((T-1, G))

    tasks.sort(key=lambda x: x[1])
    total_moves = 0
    persons = deque(X)

    for T, G in tasks:
        person = persons[T]
        moves = abs(person - G)
        total_moves += moves
        persons.rotate(-1)

    return total_moves

print(solve())