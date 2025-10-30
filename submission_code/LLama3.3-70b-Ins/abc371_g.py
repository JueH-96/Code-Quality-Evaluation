import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def read_int():
    return int(sys.stdin.readline())

def solve():
    n = read_int()
    p = [0] + read_ints()
    a = [0] + read_ints()

    # Find cycles in the permutation P
    cycles = []
    visited = [False] * (n + 1)
    for i in range(1, n + 1):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = p[j]
            cycles.append(cycle)

    # Find the lexicographically smallest A that can be obtained
    min_a = [0] * (n + 1)
    for cycle in cycles:
        min_val = float('inf')
        for i in cycle:
            if a[i] < min_val:
                min_val = a[i]
        for i in cycle:
            min_a[i] = min_val

    # Print the lexicographically smallest A
    print(' '.join(map(str, min_a[1:])))

if __name__ == '__main__':
    solve()