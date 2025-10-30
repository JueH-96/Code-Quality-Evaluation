# YOUR CODE HERE
import itertools

def solve():
    n, m = map(int, input().split())
    sets = []
    for _ in range(n):
        a = int(input())
        s = list(map(int, input().split()))
        sets.append(set(s))

    def can_merge(s1, s2):
        return len(s1.intersection(s2)) > 0

    def merge(s1, s2):
        return s1.union(s2)

    q = [(sets, 0)]
    visited = {tuple(tuple(s) for s in sets)}

    while q:
        current_sets, count = q.pop(0)

        for i in range(len(current_sets)):
            for j in range(i + 1, len(current_sets)):
                if can_merge(current_sets[i], current_sets[j]):
                    new_sets = current_sets[:i] + current_sets[i+1:j] + current_sets[j+1:] + [merge(current_sets[i], current_sets[j])]
                    
                    new_sets_tuple = tuple(tuple(sorted(s)) for s in new_sets)
                    if new_sets_tuple not in visited:
                        visited.add(new_sets_tuple)
                        
                        for s in new_sets:
                            if 1 in s and m in s:
                                print(count + 1)
                                return
                        q.append((new_sets, count + 1))
    
    for s in sets:
        if 1 in s and m in s:
            print(0)
            return

    print(-1)

solve()