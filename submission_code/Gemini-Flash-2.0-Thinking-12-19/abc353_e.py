import collections

def solve():
    n = int(input())
    strings = input().split()
    if n < 2:
        print(0)
        return
    
    trie = {}
    node_counts = collections.defaultdict(int)
    
    for s in strings:
        current_node = trie
        node_counts[id(current_node)] += 1
        for char in s:
            if char not in current_node:
                current_node[char] = {}
            current_node = current_node[char]
            node_counts[id(current_node)] += 1
            
    total_lcp_sum = 0
    
    for node_id in node_counts:
        count = node_counts[node_id]
        if count >= 2:
            total_lcp_sum += count * (count - 1) // 2
            
    root_count = node_counts[id(trie)] if id(trie) in node_counts else 0
    if root_count >= 2:
        total_lcp_sum -= root_count * (root_count - 1) // 2 # Subtract root contribution
        
    depth_sum = 0
    
    def get_depth_sum(node, current_depth):
        nonlocal depth_sum
        node_id = id(node)
        if node_id in node_counts:
            count = node_counts[node_id]
            if current_depth > 0 and count >= 2:
                depth_sum += (count * (count - 1) // 2)
        for char in node:
            get_depth_sum(node[char], current_depth + 1)
            
    get_depth_sum(trie, 0)
    
    root_node_count = node_counts[id(trie)] if id(trie) in node_counts else 0
    if root_node_count >= 2:
        depth_sum -= (root_node_count * (root_node_count - 1) // 2)

    print(depth_sum)

if __name__ == '__main__':
    solve()