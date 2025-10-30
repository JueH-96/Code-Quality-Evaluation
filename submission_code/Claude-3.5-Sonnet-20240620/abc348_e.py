# YOUR CODE HERE
from collections import defaultdict
import sys

def dfs(node, parent, depth, tree, depths):
    depths[node] = depth
    for child in tree[node]:
        if child != parent:
            dfs(child, node, depth + 1, tree, depths)

def rerooting(node, parent, tree, depths, costs, subtree_sum, subtree_cost, total_sum, ans):
    current_ans = subtree_cost + (total_sum - subtree_sum)
    ans[0] = min(ans[0], current_ans)

    for child in tree[node]:
        if child != parent:
            child_sum = sum(costs[v] for v in tree[child] if v != node)
            child_cost = sum(costs[v] * (depths[v] - depths[child] + 1) for v in tree[child] if v != node)
            new_subtree_sum = subtree_sum - costs[child] + (total_sum - subtree_sum)
            new_subtree_cost = subtree_cost - child_cost - costs[child] + (total_sum - subtree_sum) + (child_sum + costs[node])
            rerooting(child, node, tree, depths, costs, new_subtree_sum, new_subtree_cost, total_sum, ans)

def solve():
    N = int(input())
    tree = defaultdict(list)
    for _ in range(N - 1):
        a, b = map(int, input().split())
        tree[a].append(b)
        tree[b].append(a)
    
    costs = [0] + list(map(int, input().split()))
    
    depths = [0] * (N + 1)
    dfs(1, 0, 0, tree, depths)
    
    total_sum = sum(costs)
    subtree_sum = total_sum
    subtree_cost = sum(costs[i] * depths[i] for i in range(1, N + 1))
    
    ans = [float('inf')]
    rerooting(1, 0, tree, depths, costs, subtree_sum, subtree_cost, total_sum, ans)
    
    print(ans[0])

solve()