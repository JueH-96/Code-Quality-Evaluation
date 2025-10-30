import sys
from collections import deque

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    p_list = [int(next(it)) for _ in range(n-1)]
    
    children = [[] for _ in range(n+1)]
    for idx in range(2, n+1):
        parent = p_list[idx-2]
        children[parent].append(idx)
    
    max_ins = [-10**18] * (n+1)
    for _ in range(m):
        x = int(next(it)); y = int(next(it))
        if y > max_ins[x]:
            max_ins[x] = y

    rem = [-10**18] * (n+1)
    count_covered = 0
    q = deque([1])
    rem[1] = max_ins[1]
    if rem[1] >= 0:
        count_covered += 1
        
    while q:
        u = q.popleft()
        for v in children[u]:
            candidate1 = rem[u] - 1
            candidate2 = max_ins[v]
            if candidate1 > candidate2:
                rem[v] = candidate1
            else:
                rem[v] = candidate2
                
            if rem[v] >= 0:
                count_covered += 1
            q.append(v)
            
    print(count_covered)

if __name__ == "__main__":
    main()