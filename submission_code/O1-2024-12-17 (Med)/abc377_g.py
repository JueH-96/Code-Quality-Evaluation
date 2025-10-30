def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]
    
    # We will build a trie (prefix tree) incrementally. For each k from 1..N:
    # 1) We query S_k by traversing in the trie to find the deepest matching prefix.
    #    Let the minimum value stored along the matched path be bestVal.
    #    The cost to transform S_k into some S_j (j<k) is ( |S_k| + bestVal ).
    #    We also compare with the cost to transform S_k into the empty string (which is |S_k|).
    #    So final answer for k is min( |S_k|, |S_k| + bestVal ).
    #    If we never find any S_j (i.e. trie is effectively empty or no prefix match),
    #    bestVal remains large, so the cost becomes min(|S_k|, |S_k| + large) = |S_k|.
    # 2) Then we insert S_k into the trie. For a node at depth d in the path of S_k,
    #    we update nodeMinVal[node] = min( nodeMinVal[node], |S_k| - 2*d ).

    # The trie node structure:
    #   - ch : dict (mapping from character index [0..25] to nextNode)
    #   - minVal : int (stores the minimum of (|someString| - 2*depth) among all strings passing here)
    
    sys.setrecursionlimit(10**7)
    
    class TrieNode:
        __slots__ = ('ch', 'minVal')
        def __init__(self):
            self.ch = {}
            self.minVal = 10**15  # A large initial value
    
    # Our trie will be a list of TrieNode objects.
    # node 0 = root
    nodes = [TrieNode()]  # start with one node, index 0
    # function to create a new trie node
    def create_node():
        nodes.append(TrieNode())
        return len(nodes) - 1
    
    # Insert a string s into the trie
    def insert_string(s):
        node = 0
        L = len(s)
        # depth=0 => we are at the root
        # update minVal at the root
        nodes[node].minVal = min(nodes[node].minVal, L - 2*0)
        depth = 0
        for c in s:
            cInd = ord(c) - ord('a')
            if cInd not in nodes[node].ch:
                new_node = create_node()
                nodes[node].ch[cInd] = new_node
            node = nodes[node].ch[cInd]
            depth += 1
            # here the depth is exactly how many characters we've consumed
            nodes[node].minVal = min(nodes[node].minVal, L - 2*depth)
    
    # Query a string s in the trie: find the minimal nodeMinVal along the deepest prefix
    def query_string(s):
        node = 0
        best = nodes[node].minVal  # check root's minVal
        for c in s:
            cInd = ord(c) - ord('a')
            if cInd not in nodes[node].ch:
                break
            node = nodes[node].ch[cInd]
            if nodes[node].minVal < best:
                best = nodes[node].minVal
        return best
    
    # We process strings S_1..S_N in order:
    # At step k, we:
    #   1) do bestVal = query_string(S_k)
    #   2) cost = min(|S_k|, |S_k| + bestVal)
    #   3) print cost
    #   4) insert_string(S_k)
    
    out = []
    for k in range(N):
        t = S[k]
        L = len(t)
        bestVal = query_string(t)
        # cost if we transform T into empty
        cost_empty = L
        # cost if we transform T into some S_j, j<k
        cost_match = L + bestVal  # because cost = |T| + (|S_j| - 2LCP)
        ans = min(cost_empty, cost_match)
        out.append(str(ans))
        
        # now insert S_k
        insert_string(t)
    
    print("
".join(out))

# Call main() to ensure the solution is executed.
main()