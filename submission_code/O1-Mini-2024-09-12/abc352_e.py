# YOUR CODE HERE
import sys
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from sys import stdin
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    operations = []
    for _ in range(M):
        K = int(data[idx]); idx +=1
        C = int(data[idx]); idx +=1
        S = list(map(int, data[idx:idx+K]))
        idx +=K
        operations.append( (C, S) )
    operations.sort()
    parent = [i for i in range(N+1)]
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    total =0
    component_count = N
    for C, S in operations:
        unique_parents = []
        seen = set()
        for u in S:
            pu = find(u)
            if pu not in seen:
                seen.add(pu)
                unique_parents.append(pu)
        k = len(unique_parents)
        if k >1:
            total += (k-1)*C
            leader = unique_parents[0]
            for p in unique_parents[1:]:
                if parent[p] != leader:
                    parent[p] = leader
                    component_count -=1
    if component_count ==1:
        print(total)
    else:
        print(-1)

if __name__ == "__main__":
    main()