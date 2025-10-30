import collections

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    edges = collections.defaultdict(list)
    for i in range(n):
        for j in range(a[i] + 1, i):
            edges[i+1].append(j+1)
        if a[i] > 0:
            edges[a[i]].append(i+1)
            
    memo = {}
    
    def get_source_nodes(nodes_set, current_edges):
        in_degree = collections.defaultdict(int)
        nodes_list = sorted(list(nodes_set))
        for u in nodes_list:
            for v in current_edges[u]:
                if v in nodes_set:
                    in_degree[v] += 1
        source_nodes = []
        for node in nodes_list:
            if in_degree[node] == 0:
                source_nodes.append(node)
        return source_nodes

    def count_topological_sorts(remaining_nodes_tuple):
        remaining_nodes_set = frozenset(remaining_nodes_tuple)
        if not remaining_nodes_set:
            return 1
        if remaining_nodes_set in memo:
            return memo[remaining_nodes_set]
        
        source_nodes = get_source_nodes(remaining_nodes_set, edges)
        count = 0
        for source_node in source_nodes:
            next_nodes_tuple = tuple(sorted(list(remaining_nodes_set - {source_node})))
            count = (count + count_topological_sorts(next_nodes_tuple)) % 998244353
            
        memo[remaining_nodes_set] = count
        return count
        
    initial_nodes = tuple(range(1, n + 1))
    result = count_topological_sorts(initial_nodes)
    print(result)

if __name__ == '__main__':
    solve()