# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from sys import stdin
    from collections import defaultdict

    input = sys.stdin.read().split()
    idx =0
    N=int(input[idx]); idx+=1
    adj = [[] for _ in range(N+1)]
    degrees = [0]*(N+1)
    for _ in range(N-1):
        u=int(input[idx]); idx+=1
        v=int(input[idx]); idx+=1
        adj[u].append(v)
        adj[v].append(u)
        degrees[u]+=1
        degrees[v]+=1

    total_leaves = [0]*(N+1)
    for node in range(1,N+1):
        count=0
        for neighbor in adj[node]:
            if degrees[neighbor]==1:
                count+=1
        total_leaves[node]=count

    max_size=0
    for node in range(1,N+1):
        A=[]
        for child in adj[node]:
            if degrees[node]==1:
                # Check if node is a leaf connected to child
                # Since degrees[node]==1, node is a leaf connected to child
                leaf_cnt = total_leaves[child] -1
            else:
                leaf_cnt = total_leaves[child]
            if leaf_cnt >=1:
                A.append(leaf_cnt)
        if A:
            A.sort(reverse=True)
            for i, y in enumerate(A):
                x = i +1
                size =1 +x +x*y
                if size > max_size:
                    max_size=size
    # The problem states it's always possible to form a snowflake, which minimally requires 3 nodes
    # But to handle cases where max_size might be less, ensure at least 3
    if max_size <3:
        max_size=3
    # However, ensure max_size does not exceed N
    if max_size >N:
        max_size=N
    print(N - max_size)

if __name__ == "__main__":
    main()