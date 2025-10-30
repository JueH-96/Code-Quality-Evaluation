def main():
    import sys
    from collections import deque

    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    prerequisites = [[] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        line = list(map(int, data[i].split()))
        C_i = line[0]
        prerequisites[i] = line[1:C_i + 1]
    
    # To find the books to read to reach book 1
    to_read = []
    visited = [False] * (N + 1)
    queue = deque([1])
    
    while queue:
        current = queue.popleft()
        if visited[current]:
            continue
        visited[current] = True
        
        for pre in prerequisites[current]:
            if not visited[pre]:
                to_read.append(pre)
                queue.append(pre)
    
    # The order should be the reverse of the order we found them
    to_read.reverse()
    
    print(" ".join(map(str, to_read)))

if __name__ == "__main__":
    main()