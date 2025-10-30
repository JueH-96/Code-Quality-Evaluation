import collections

def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))
    
    squared_d = d * d
    adjacency_list = collections.defaultdict(list)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq <= squared_d:
                adjacency_list[i+1].append(j+1)
                adjacency_list[j+1].append(i+1)
                
    is_infected = [False] * (n + 1)
    is_infected[1] = True
    
    queue = collections.deque([1])
    
    while queue:
        person_index = queue.popleft()
        for neighbor_index in adjacency_list[person_index]:
            if not is_infected[neighbor_index]:
                is_infected[neighbor_index] = True
                queue.append(neighbor_index)
                
    output = []
    for i in range(1, n + 1):
        if is_infected[i]:
            output.append("Yes")
        else:
            output.append("No")
            
    for result in output:
        print(result)

if __name__ == '__main__':
    solve()