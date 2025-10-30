import sys

def find_cycle(N, A):
    visited = [False] * (N + 1)
    cycle = []
    
    for i in range(1, N + 1):
        if not visited[i]:
            current = i
            path = []
            while not visited[current]:
                visited[current] = True
                path.append(current)
                current = A[current - 1]
                if current in path:
                    cycle_start = path.index(current)
                    cycle = path[cycle_start:]
                    break
            if cycle:
                break
    return cycle

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    cycle = find_cycle(N, A)
    print(len(cycle))
    print(' '.join(map(str, cycle)))

if __name__ == "__main__":
    main()