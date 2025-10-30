import sys
from collections import deque, defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(input[idx])
        b = int(input[idx + 1])
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    
    degree = [0] * (N + 1)
    for i in range(1, N + 1):
        degree[i] = len(adj[i])
    
    leaves = deque()
    parent_to_leaves = defaultdict(list)
    possible_siblings = set()
    
    for i in range(1, N + 1):
        if degree[i] == 1:
            leaves.append(i)
            p = None
            for neighbor in adj[i]:
                if degree[neighbor] > 0:
                    p = neighbor
                    break
            parent_to_leaves[p].append(i)
            if len(parent_to_leaves[p]) >= 2:
                possible_siblings.add(p)
    
    while leaves:
        if possible_siblings:
            p = possible_siblings.pop()
            u = parent_to_leaves[p].pop()
            v = parent_to_leaves[p].pop()
            if len(parent_to_leaves[p]) < 2:
                possible_siblings.discard(p)
            print(u, v)
            # Update degrees for u's neighbors
            for neighbor in adj[u]:
                if degree[neighbor] > 0:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        leaves.append(neighbor)
                        parent = None
                        for nbr in adj[neighbor]:
                            if degree[nbr] > 0:
                                parent = nbr
                                break
                        parent_to_leaves[parent].append(neighbor)
                        if len(parent_to_leaves[parent]) >= 2:
                            possible_siblings.add(parent)
            # Update degrees for v's neighbors
            for neighbor in adj[v]:
                if degree[neighbor] > 0:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        leaves.append(neighbor)
                        parent = None
                        for nbr in adj[neighbor]:
                            if degree[nbr] > 0:
                                parent = nbr
                                break
                        parent_to_leaves[parent].append(neighbor)
                        if len(parent_to_leaves[parent]) >= 2:
                            possible_siblings.add(parent)
            degree[u] = 0
            degree[v] = 0
        else:
            u = leaves.popleft()
            p = None
            for neighbor in adj[u]:
                if degree[neighbor] > 0:
                    p = neighbor
                    break
            found_v = False
            for q in adj[p]:
                if degree[q] == 0:
                    continue
                if len(parent_to_leaves[q]) >= 1:
                    v = parent_to_leaves[q].pop()
                    if len(parent_to_leaves[q]) < 2:
                        possible_siblings.discard(q)
                    print(u, v)
                    # Update for u's neighbors
                    for neighbor in adj[u]:
                        if degree[neighbor] > 0:
                            degree[neighbor] -= 1
                            if degree[neighbor] == 1:
                                leaves.append(neighbor)
                                parent = None
                                for nbr in adj[neighbor]:
                                    if degree[nbr] > 0:
                                        parent = nbr
                                        break
                                parent_to_leaves[parent].append(neighbor)
                                if len(parent_to_leaves[parent]) >= 2:
                                    possible_siblings.add(parent)
                    # Update for v's neighbors
                    for neighbor in adj[v]:
                        if degree[neighbor] > 0:
                            degree[neighbor] -= 1
                            if degree[neighbor] == 1:
                                leaves.append(neighbor)
                                parent = None
                                for nbr in adj[neighbor]:
                                    if degree[nbr] > 0:
                                        parent = nbr
                                        break
                                parent_to_leaves[parent].append(neighbor)
                                if len(parent_to_leaves[parent]) >= 2:
                                    possible_siblings.add(parent)
                    degree[u] = 0
                    degree[v] = 0
                    found_v = True
                    break
            if found_v:
                continue
            # Pair u with p
            print(u, p)
            # Update for u's neighbors
            for neighbor in adj[u]:
                if degree[neighbor] > 0:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        leaves.append(neighbor)
                        parent = None
                        for nbr in adj[neighbor]:
                            if degree[nbr] > 0:
                                parent = nbr
                                break
                        parent_to_leaves[parent].append(neighbor)
                        if len(parent_to_leaves[parent]) >= 2:
                            possible_siblings.add(parent)
            # Update for p's neighbors
            for neighbor in adj[p]:
                if degree[neighbor] > 0:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        leaves.append(neighbor)
                        parent = None
                        for nbr in adj[neighbor]:
                            if degree[nbr] > 0:
                                parent = nbr
                                break
                        parent_to_leaves[parent].append(neighbor)
                        if len(parent_to_leaves[parent]) >= 2:
                            possible_siblings.add(parent)
            degree[u] = 0
            degree[p] = 0

if __name__ == "__main__":
    main()