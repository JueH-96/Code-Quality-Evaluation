def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    strings = data[1:N+1]
    strings.sort()
    
    trie = [[0, [None]*26]]  # root node: [count, children]
    
    for s in strings:
        node = 0  # start from root
        for c in s:
            idx = ord(c) - ord('a')
            if trie[node][1][idx] is None:
                trie.append([0, [None]*26])
                trie[node][1][idx] = len(trie) - 1
            node = trie[node][1][idx]
            trie[node][0] += 1
    
    total = 0
    for node in range(1, len(trie)):
        m = trie[node][0]
        if m >= 2:
            total += m * (m - 1) // 2
    
    print(total)

if __name__ == "__main__":
    main()