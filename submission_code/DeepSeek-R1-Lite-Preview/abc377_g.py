class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.max_val = -float('inf')

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    strings = data[1:N+1]
    
    root = TrieNode()
    
    for k in range(N):
        Sk = strings[k]
        m = len(Sk)
        
        # Find the maximum (2L - |Sj|) over j < k for suffixes of Sk
        current = root
        max_collected = -float('inf')
        for i in range(m-1, -1, -1):
            char = Sk[i]
            idx = ord(char) - ord('a')
            if current.children[idx] is None:
                break  # No previous Sj has this suffix
            current = current.children[idx]
            # L is m - i (length of the suffix)
            # But since we stored 2L - |Sj| in max_val
            max_collected = max(max_collected, current.max_val)
        
        # Calculate min_cost_k
        min_cost_k = min(m, m - max_collected)
        print(min_cost_k)
        
        # Insert Sk into the trie, updating max_val for each suffix
        current = root
        for i in range(m-1, -1, -1):
            char = Sk[i]
            idx = ord(char) - ord('a')
            if current.children[idx] is None:
                current.children[idx] = TrieNode()
            current = current.children[idx]
            L = m - i
            val = 2 * L - m
            if current.max_val < val:
                current.max_val = val

if __name__ == "__main__":
    main()