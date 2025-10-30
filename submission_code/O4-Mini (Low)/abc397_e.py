import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    
    input = sys.stdin.readline
    N, K = map(int, input().split())
    NK = N * K
    
    if K == 1:
        # Every vertex is a path of length 1 trivially.
        print("Yes")
        return
    
    # Read tree
    adj = [[] for _ in range(NK+1)]
    degree = [0]*(NK+1)
    for _ in range(NK-1):
        u,v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    # up[v]: length of the partial path ending at v coming from its removed subtree
    up = [0]*(NK+1)
    removed = [False]*(NK+1)
    
    q = deque()
    # initialize leaves
    for v in range(1, NK+1):
        if degree[v] <= 1:
            q.append(v)
            up[v] = 1  # single vertex path
    
    removed_count = 0
    while q:
        v = q.popleft()
        if removed[v]:
            continue
        # if we've built a partial path of length K at v, we close it here
        if up[v] == K:
            # close path: remove this whole chain of K vertices
            # but actually we only need to mark v removed and not propagate
            removed[v] = True
            removed_count += 1
            # reduce degree of neighbors
            for w in adj[v]:
                if not removed[w]:
                    degree[w] -= 1
                    if degree[w] == 1:
                        # start new leaf with up=1
                        up[w] = 1
                        q.append(w)
            continue
        
        # otherwise we must propagate up[v]+1 to its unique neighbor
        # but if v has no neighbor left => fail
        # or if up[v] > K => fail
        if up[v] > K:
            print("No")
            return
        
        # find the unique unremoved neighbor
        parent = -1
        for w in adj[v]:
            if not removed[w]:
                parent = w
                break
        if parent == -1:
            # no place to propagate but we haven't reached length K
            print("No")
            return
        
        # remove v and propagate
        removed[v] = True
        removed_count += 1
        degree[v] = 0
        degree[parent] -= 1
        # new partial path length at parent
        up[parent] = max(up[parent], up[v] + 1)
        
        # if parent becomes a leaf, enqueue
        if degree[parent] == 1:
            q.append(parent)
    
    # if we've removed all vertices successfully, answer is Yes
    print("Yes" if removed_count == NK else "No")

if __name__ == "__main__":
    main()