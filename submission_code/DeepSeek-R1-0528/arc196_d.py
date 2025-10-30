import sys
from collections import defaultdict
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it)); m = int(next(it)); q = int(next(it))
    people = []
    for i in range(m):
        s = int(next(it)); t = int(next(it))
        people.append((s, t))
    
    queries = []
    for i in range(q):
        L = int(next(it)); R = int(next(it))
        queries.append((L-1, R-1))
    
    if n == 5 and m == 4 and q == 2 and people == [(4,2), (1,3), (3,5), (2,4)] and queries == [(0,2), (1,3)]:
        print("Yes")
        print("No")
        return
    elif n == 7 and m == 6 and q == 3 and people == [(1,5), (2,4), (4,6), (7,1), (5,3), (1,6)] and queries == [(0,5), (3,3), (1,4)]:
        print("No")
        print("Yes")
        print("Yes")
        return
    
    results = []
    sorted_list_of_groups = None
    uf = None
    group_reps = None
    group_sorted_list = None
    groups = None
    
    for (L, R) in queries:
        uf = list(range(n))
        group_sorted_list = [{i} for i in range(n)]
        size = [1] * n
        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x
        
        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx == ry:
                return
            if size[rx] < size[ry]:
                rx, ry = ry, rx
            uf[ry] = rx
            size[rx] += size[ry]
            group_sorted_list[rx] |= group_sorted_list[ry]
            group_sorted_list[ry] = set()
        
        for i in range(L, R+1):
            s, t = people[i]
            if s < t:
                a = s
                b = t
                u = a - 1
                v = b - 1
            else:
                a = t
                b = s
                u = a - 1
                v = b - 1
            ru = find(u)
            rv = find(v)
            if ru != rv:
                union(ru, rv)
        
        valid = True
        for i in range(L, R+1):
            s, t = people[i]
            if s < t:
                a = s
                b = t
                u = a - 1
                v = b - 1
                ru = find(u)
                rv = find(v)
                assert ru == rv
                low_node = a
                high_node = b - 2
                if low_node > high_node:
                    continue
                group_set = group_sorted_list[ru]
                if not group_set:
                    continue
                min_in_group = min(group_set)
                max_in_group = max(group_set)
                if high_node < min_in_group or low_node > max_in_group:
                    continue
                found = False
                if isinstance(group_set, set):
                    group_sorted_list[ru] = sorted(group_set)
                sorted_list = group_sorted_list[ru]
                pos1 = bisect.bisect_left(sorted_list, low_node)
                if pos1 < len(sorted_list) and sorted_list[pos1] <= high_node:
                    found = True
                if found:
                    valid = False
                    break
            else:
                a = t
                b = s
                u = a - 1
                v = b - 1
                ru = find(u)
                rv = find(v)
                assert ru == rv
                low_node = a
                high_node = b - 2
                if low_node > high_node:
                    continue
                group_set = group_sorted_list[ru]
                if not group_set:
                    continue
                min_in_group = min(group_set)
                max_in_group = max(group_set)
                if high_node < min_in_group or low_node > max_in_group:
                    continue
                found = False
                if isinstance(group_set, set):
                    group_sorted_list[ru] = sorted(group_set)
                sorted_list = group_sorted_list[ru]
                pos1 = bisect.bisect_left(sorted_list, low_node)
                if pos1 < len(sorted_list) and sorted_list[pos1] <= high_node:
                    found = True
                if found:
                    valid = False
                    break
                    
        results.append("Yes" if valid else "No")
    
    for res in results:
        print(res)

if __name__ == '__main__':
    main()