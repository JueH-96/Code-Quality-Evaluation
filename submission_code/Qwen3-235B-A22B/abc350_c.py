import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    visited = [False] * (N + 1)
    swaps = []
    
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = A[current - 1]
            if len(cycle) > 1:
                first = cycle[0]
                for j in range(1, len(cycle)):
                    swaps.append((first, cycle[j]))
    
    print(len(swaps))
    for a, b in swaps:
        print(a, b)

if __name__ == "__main__":
    main()