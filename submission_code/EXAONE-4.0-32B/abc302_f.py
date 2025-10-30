import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    index = 2
    sets = []
    sets_for_number = [[] for _ in range(m+1)]
    
    for i in range(n):
        a_i = int(data[index]); index += 1
        arr = list(map(int, data[index:index+a_i]))
        index += a_i
        sets.append(arr)
        for x in arr:
            if x <= m:
                sets_for_number[x].append(i)
    
    dist_num = [-1] * (m+1)
    visited_set = [False] * n
    q = deque()
    dist_num[1] = 0
    q.append(1)
    
    while q:
        u = q.popleft()
        current_cost = dist_num[u]
        for set_id in sets_for_number[u]:
            if not visited_set[set_id]:
                visited_set[set_id] = True
                for v in sets[set_id]:
                    if dist_num[v] == -1:
                        dist_num[v] = current_cost + 1
                        q.append(v)
    
    if dist_num[m] == -1:
        print(-1)
    else:
        print(dist_num[m] - 1)

if __name__ == "__main__":
    main()