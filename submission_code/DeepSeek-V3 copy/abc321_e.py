import sys
from collections import deque

def count_nodes_at_distance(N, X, K):
    if K == 0:
        return 1
    visited = set()
    queue = deque()
    queue.append((X, 0))
    visited.add(X)
    count = 0
    while queue:
        current, dist = queue.popleft()
        if dist == K:
            count += 1
            continue
        # Parent
        parent = current // 2
        if parent >= 1 and parent not in visited and dist + 1 <= K:
            visited.add(parent)
            queue.append((parent, dist + 1))
        # Left child
        left_child = 2 * current
        if left_child <= N and left_child not in visited and dist + 1 <= K:
            visited.add(left_child)
            queue.append((left_child, dist + 1))
        # Right child
        right_child = 2 * current + 1
        if right_child <= N and right_child not in visited and dist + 1 <= K:
            visited.add(right_child)
            queue.append((right_child, dist + 1))
    return count

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        N = int(data[index])
        X = int(data[index+1])
        K = int(data[index+2])
        index += 3
        if K == 0:
            print(1)
            continue
        if K > 60:
            print(0)
            continue
        # Calculate the number of nodes at distance K
        # Using BFS is not efficient for large N, so we need a mathematical approach
        # The tree is a binary tree, so we can calculate the number of nodes at distance K
        # by considering the levels of the tree
        # The number of nodes at distance K is the number of nodes at level (depth of X + K)
        # minus the nodes that are not in the tree
        # But since N can be up to 10^18, we need a smarter way
        # We can use the fact that the tree is a complete binary tree up to level log2(N)
        # So we can calculate the number of nodes at distance K by considering the levels
        # and the boundaries of the tree
        # For now, we will use the BFS approach for small K and N
        if K <= 20 and N <= 10**6:
            print(count_nodes_at_distance(N, X, K))
        else:
            # For large K and N, we need a mathematical approach
            # The number of nodes at distance K is the number of nodes in the K-th level of the tree
            # starting from X
            # The K-th level has 2^K nodes, but we need to check if they are within the tree
            # So the number of nodes is min(2^K, number of nodes in the K-th level starting from X)
            # To find the number of nodes in the K-th level starting from X, we can calculate the range
            # of nodes in that level
            # The first node in the K-th level is X * 2^K
            # The last node is X * 2^K + 2^K - 1
            # So the number of nodes is min(2^K, N - X * 2^K + 1)
            # But we need to ensure that X * 2^K <= N
            lower = X * (1 << K)
            upper = lower + (1 << K) - 1
            if lower > N:
                print(0)
            else:
                print(min(1 << K, N - lower + 1))

if __name__ == "__main__":
    main()