import sys

def find_cycle(n, edges):
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        cycle = []
        current = i
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = edges[current] - 1
        if cycle[0] == edges[current] - 1:
            return cycle
    return []

def main():
    n = int(input().strip())
    edges = list(map(int, input().strip().split()))
    cycle = find_cycle(n, edges)
    print(len(cycle))
    print(' '.join(str(v + 1) for v in cycle))

if __name__ == "__main__":
    main()