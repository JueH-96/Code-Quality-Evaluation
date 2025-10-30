def main():
    N = int(input())
    A = list(map(int, input().split()))
    A = [a-1 for a in A]  # 0-indexed
    visited = [0]*N
    current = 0
    cycle = []
    while True:
        if visited[current]:
            cycle = cycle[cycle.index(current):]
            break
        else:
            visited[current] = 1
            cycle.append(current)
            current = A[current]
    print(len(cycle))
    for v in cycle:
        print(v+1)

if __name__ == "__main__":
    main()