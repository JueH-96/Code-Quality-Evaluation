from collections import defaultdict

def solve():
    N = int(input())
    used_names = set()
    name_count = defaultdict(int)
    for _ in range(N):
        name = input()
        used_names.add(name)
        name_count[name] += 1

    ng_list = []
    for name in used_names:
        found = False
        for i in range(len(name)):
            for j in range(i+1, len(name)+1):
                substr = name[i:j]
                if substr in used_names and name_count[substr] == 1:
                    found = True
                    break
            if found:
                break
        if not found:
            ng_list.append(name)

    return len(ng_list)

print(solve())