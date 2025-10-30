# YOUR CODE HERE
import sys

def find_cycle(n, a):
    visited = [False] * (n + 1)
    path = [-1] * (n + 1)
    current = 1

    while not visited[current]:
        visited[current] = True
        path[current] = len(path) - 1
        current = a[current - 1]

    start = path[current]
    cycle = path[start + 1:] + path[:start + 1]
    return len(cycle), cycle

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:]))
    cycle_length, cycle = find_cycle(n, a)
    print(cycle_length)
    print(' '.join(map(str, cycle)))

if __name__ == "__main__":
    main()