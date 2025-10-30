# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr +=2
    sets = []
    element_to_sets = [[] for _ in range(M+1)]
    sets_contain_1 = []
    sets_contain_M = []
    for i in range(N):
        Ai = int(input[ptr])
        ptr +=1
        elems = list(map(int, input[ptr:ptr+Ai]))
        ptr +=Ai
        sets.append(elems)
        for e in elems:
            element_to_sets[e].append(i)
        if 1 in elems:
            sets_contain_1.append(i)
        if M in elems:
            sets_contain_M.append(i)
    # Check if any set contains both 1 and M
    for s in sets_contain_1:
        if M in sets[s]:
            print(0)
            return
    if not sets_contain_1 or not sets_contain_M:
        print(-1)
        return
    visited = [False]*N
    q = deque()
    for s in sets_contain_1:
        visited[s] = True
        q.append((s,0))
    element_processed = [False]*(M+1)
    while q:
        current, dist = q.popleft()
        # Check if current set contains M
        if M in sets[current]:
            print(dist)
            return
        for e in sets[current]:
            if not element_processed[e]:
                for s in element_to_sets[e]:
                    if not visited[s]:
                        visited[s] = True
                        q.append((s, dist+1))
                element_processed[e] = True
    print(-1)

if __name__ == "__main__":
    main()