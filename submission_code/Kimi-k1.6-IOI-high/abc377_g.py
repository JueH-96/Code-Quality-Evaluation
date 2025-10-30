class TrieNode:
    __slots__ = ['children', 'min_val']
    def __init__(self):
        self.children = dict()
        self.min_val = float('inf')

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = []
    for _ in range(N):
        S.append(input[idx])
        idx +=1
    
    root = TrieNode()
    
    for s in S:
        # Compute answer
        current_node = root
        min_cost = len(s)
        # Check m=0 (root node)
        if current_node.min_val != float('inf'):
            candidate = len(s) + current_node.min_val
            if candidate < min_cost:
                min_cost = candidate
        
        # Traverse each character
        depth = 0
        for c in s:
            if c not in current_node.children:
                break
            current_node = current_node.children[c]
            depth += 1
            if current_node.min_val != float('inf'):
                candidate = len(s) + current_node.min_val
                if candidate < min_cost:
                    min_cost = candidate
        
        print(min_cost)
        
        # Insert the current string into the trie
        current_node = root
        current_node.min_val = min(current_node.min_val, len(s) - 0 * 2)
        depth = 0
        for c in s:
            if c not in current_node.children:
                current_node.children[c] = TrieNode()
            current_node = current_node.children[c]
            depth += 1
            val = len(s) - 2 * depth
            if val < current_node.min_val:
                current_node.min_val = val

if __name__ == "__main__":
    main()