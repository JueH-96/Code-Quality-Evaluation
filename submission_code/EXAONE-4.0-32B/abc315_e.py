import heapq
import collections
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    prereq = [[] for _ in range(n+1)]
    index = 1
    for i in range(1, n+1):
        c_i = int(data[index]); index += 1
        if c_i > 0:
            prereq[i] = list(map(int, data[index:index+c_i]))
            index += c_i
            
    needed = set()
    queue = collections.deque([1])
    needed.add(1)
    while queue:
        u = queue.popleft()
        for v in prereq[u]:
            if v not in needed:
                needed.add(v)
                queue.append(v)
                
    rev_graph = [[] for _ in range(n+1)]
    in_degree = [0] * (n+1)
    
    for u in needed:
        for v in prereq[u]:
            rev_graph[v].append(u)
        in_degree[u] = len(prereq[u])
        
    heap = []
    for u in needed:
        if in_degree[u] == 0:
            heapq.heappush(heap, u)
            
    result = []
    while heap:
        u = heapq.heappop(heap)
        if u != 1:
            result.append(str(u))
        for v in rev_graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(heap, v)
                
    print(" ".join(result))

if __name__ == "__main__":
    main()