import sys
from collections import deque, defaultdict

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    parents = [0] * (N + 1)
    for i in range(2, N + 1):
        parents[i] = int(data[index])
        index += 1

    insurances = []
    for _ in range(M):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        insurances.append((x, y))

    # Build the tree
    tree = defaultdict(list)
    for child, parent in enumerate(parents[2:], start=2):
        tree[parent].append(child)

    # BFS to find all descendants within y generations
    def bfs(start, generations):
        queue = deque([(start, 0)])
        visited = set()
        while queue:
            current, gen = queue.popleft()
            if gen > generations:
                break
            visited.add(current)
            for neighbor in tree[current]:
                if neighbor not in visited:
                    queue.append((neighbor, gen + 1))
        return visited

    covered = set()
    for x, y in insurances:
        descendants = bfs(x, y)
        covered.update(descendants)

    print(len(covered))

if __name__ == "__main__":
    main()