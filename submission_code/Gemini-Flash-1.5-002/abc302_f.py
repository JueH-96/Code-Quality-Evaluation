import collections

def solve():
    n, m = map(int, input().split())
    sets = []
    for _ in range(n):
        line = list(map(int, input().split()))
        sets.append(set(line[1:]))

    q = collections.deque([(sets, 0)])
    visited = {tuple(map(tuple, sorted(sets, key=lambda x: tuple(sorted(x)))))}

    while q:
        current_sets, dist = q.popleft()
        
        all_elements = set()
        for s in current_sets:
            all_elements.update(s)

        if 1 in all_elements and m in all_elements:
            print(dist)
            return

        for i in range(len(current_sets)):
            for j in range(i + 1, len(current_sets)):
                if len(current_sets[i].intersection(current_sets[j])) > 0:
                    new_sets = list(current_sets)
                    new_set = new_sets[i].union(new_sets[j])
                    new_sets.pop(j)
                    new_sets.pop(i)
                    new_sets.append(new_set)
                    
                    sorted_new_sets = tuple(map(tuple, sorted(new_sets, key=lambda x: tuple(sorted(x)))))
                    if sorted_new_sets not in visited:
                        visited.add(sorted_new_sets)
                        q.append((new_sets, dist + 1))

    print(-1)

solve()