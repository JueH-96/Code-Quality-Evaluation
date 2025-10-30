import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]  # 1-based indexing
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    neighbors = []
    for v in adj[1]:
        neighbors.append(v)
    
    if not neighbors:
        print(1)
        return
    
    sum_size = 0
    max_size = 0
    
    for u in neighbors:
        # Compute size of subtree rooted at u, parent is 1
        size = 0
        q = deque()
        q.append(u)
        while q:
            node = q.popleft()
            size += 1
            for v in adj[node]:
                if v != 1 and v != node:
                    # To avoid revisiting parent (1), but need to check if the parent is 1 or other nodes
                    # Wait, the parent of u is 1. For other nodes in the subtree, their parent is not 1.
                    # So, during BFS, we need to avoid going back to the parent (which is the previous node)
                    # Wait, no. The code as written will traverse all nodes in the subtree of u, excluding 1.
                    # Because when processing u, we add all its neighbors except 1. Then, for each of those neighbors, their neighbors are processed, etc.
                    # Wait, no. The current code adds all neighbors of node except 1. But node's parent in the BFS is not tracked. So this code may revisit nodes.
                    # Wait, this code is incorrect. Because when you process node u, you add all its neighbors except 1. But for example, if u is connected to v (which is not 1), then when processing v, you add all its neighbors except 1, which includes u. So u will be processed again. This leads to an infinite loop and incorrect size calculation.
                    # So the code is wrong. We need to track the parent of each node to avoid revisiting.
                    # Correct approach: in the BFS, track the parent of each node. For each node, when adding neighbors, exclude the parent.
                    # So, the compute_size function needs to be modified to track parents.
                    # Let's rewrite the BFS with parent tracking.
                    # Let's use a helper function.
                    # So, here's the corrected code:
                    # Re-implementing the BFS with parent tracking.
                    # So, for the current u, we start BFS, and for each node, we track the parent to avoid cycles.
                    # So, inside the loop for each neighbor of 1:
                    size = 0
                    q = deque()
                    q.append( (u, 1) )  # (current node, parent)
                    while q:
                        node, parent = q.popleft()
                        size +=1
                        for v in adj[node]:
                            if v != parent:
                                q.append( (v, node) )
                    # Now, this will correctly compute the size of the subtree.
                    # So, the code for size computation should be modified as above.
        # After recomputing size with parent tracking:
        sum_size += size
        if size > max_size:
            max_size = size
    
    answer = (sum_size - max_size) + 1
    print(answer)

if __name__ == '__main__':
    main()