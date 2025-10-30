import sys

def read_ints(): return map(int, input().split())
def read_int(): return int(input())

def dfs(book, graph, visited, order):
    if visited[book]:
        return
    visited[book] = True
    for next_book in graph[book]:
        dfs(next_book, graph, visited, order)
    order.append(book)

def main():
    N = read_int()
    graph = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        C, *P = read_ints()
        for p in P:
            graph[i].append(p)
    
    visited = [False] * (N + 1)
    order = []
    dfs(1, graph, visited, order)
    
    result = [book for book in order if book != 1]
    print(' '.join(map(str, result[::-1])))

if __name__ == "__main__":
    main()