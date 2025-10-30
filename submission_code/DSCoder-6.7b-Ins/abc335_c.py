import sys

def read_input():
    N, Q = map(int, sys.stdin.readline().split())
    queries = [sys.stdin.readline().split() for _ in range(Q)]
    return N, Q, queries

def solve(N, Q, queries):
    parts = [(i, 0) for i in range(1, N+1)]
    for query in queries:
        if query[0] == '1':
            direction = query[1]
            for i in range(len(parts)):
                if direction == 'R':
                    parts[i] = (parts[i][0]+1, parts[i][1])
                elif direction == 'L':
                    parts[i] = (parts[i][0]-1, parts[i][1])
                elif direction == 'U':
                    parts[i] = (parts[i][0], parts[i][1]+1)
                elif direction == 'D':
                    parts[i] = (parts[i][0], parts[i][1]-1)
        else:
            p = int(query[1])
            print(parts[p-1][0], parts[p-1][1])

N, Q, queries = read_input()
solve(N, Q, queries)