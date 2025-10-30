def main():
    import sys
    sys.setrecursionlimit(10**6)
    
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    words = data[1:]
    
    # Key observation: For any pair (x, y), f(x,y) (length of longest common prefix)
    # is the number of prefix lengths d >= 1 where x and y share the same d-length prefix.
    # Thus, the total answer is:
    #   sum_{d>=1} (number of pairs sharing a common prefix of length d).
    #
    # We can build a trie where at each node (representing a prefix of some length),
    # we count how many strings share that prefix. Then, each such node contributes
    # C(n,2) to the final sum, where n is the count. The sum over nodes (except the root)
    # gives the desired answer.
    
    # Build a trie. Each node is a dictionary with key "cnt" for count and
    # "children" as a dict mapping char to child node.
    trie = {"cnt": 0, "children": {}}
    
    for word in words:
        node = trie
        node["cnt"] += 1
        for ch in word:
            if ch not in node["children"]:
                node["children"][ch] = {"cnt": 0, "children": {}}
            node = node["children"][ch]
            node["cnt"] += 1
    
    # Traverse the trie (except the root) and sum the combinations C(cnt, 2) for each node.
    ans = 0
    stack = [trie]
    while stack:
        node = stack.pop()
        for child in node["children"].values():
            cnt = child["cnt"]
            if cnt >= 2:
                ans += cnt * (cnt - 1) // 2
            stack.append(child)
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()