s = input().strip()
t = input().strip()

if s == t:
    print(0)
else:
    x = []
    current = s
    while current != t:
        candidates = []
        for i in range(len(current)):
            if current[i] != t[i]:
                new_str = current[:i] + t[i] + current[i+1:]
                candidates.append(new_str)
        next_str = min(candidates)
        x.append(next_str)
        current = next_str
    print(len(x))
    for line in x:
        print(line)