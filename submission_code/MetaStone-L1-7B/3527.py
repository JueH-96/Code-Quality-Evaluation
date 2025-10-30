def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    colors = list(map(int, data[ptr:ptr + len(data)//4]))
    ptr += len(data)//4
    queries = data[ptr:]
    
    from collections import defaultdict
    freq = defaultdict(int)
    
    n = len(colors)
    transitions = []
    for i in range(n):
        if colors[i] != colors[(i+1) % n]:
            transitions.append(i)
    transitions.sort()
    
    def compute_groups(transitions):
        groups = []
        m = len(transitions)
        if m == 0:
            return defaultdict(int)
        for i in range(m - 1):
            groups.append((transitions[i+1] - transitions[i]) % n)
        if m > 0:
            groups.append((transitions[0] + n - transitions[-1]) % n)
        return defaultdict(int)
    
    for query in queries:
        if query == '1':
            s = int(query.split()[1])
            groups = compute_groups(transitions)
            freq[s] = groups.get(s, 0)
            print(freq[s])
        else:
            idx = int(query.split()[1])
            new_color = int(query.split()[2])
            old_color = colors[idx]
            colors[idx] = new_color
            for j in [idx - 1, idx, idx + 1]:
                j = j % n
                if j in transitions:
                    transitions.remove(j)
            for j in [idx - 1, idx, idx + 1]:
                j = j % n
                if colors[j] != colors[(j + 1) % n]:
                    if j in transitions:
                        transitions.remove(j)
                    else:
                        transitions.append(j)
            transitions.sort()
            groups = compute_groups(transitions)
            freq.clear()
            for s in groups:
                freq[s] = groups[s]
    
if __name__ == '__main__':
    main()