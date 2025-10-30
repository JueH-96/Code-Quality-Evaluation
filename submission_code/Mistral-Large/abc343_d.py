import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    T = int(data[index])
    index += 1

    events = []
    for _ in range(T):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        events.append((A, B))

    scores = [0] * (N + 1)
    unique_scores = set()
    unique_scores.add(0)
    result = []

    for i in range(T):
        player, points = events[i]
        scores[player] += points
        unique_scores.add(scores[player])
        result.append(len(unique_scores))

    for value in result:
        print(value)

if __name__ == "__main__":
    main()