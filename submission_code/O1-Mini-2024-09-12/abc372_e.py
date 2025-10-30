import sys
import sys
import sys
from sys import stdin
import sys
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    index = 0
    N, Q = int(input[index]), int(input[index+1])
    index +=2
    parent = [i for i in range(N+1)]
    size = [1]*(N+1)
    top = [[i] for i in range(N+1)]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    output = []
    for _ in range(Q):
        if input[index] == '1':
            u = int(input[index+1])
            v = int(input[index+2])
            index +=3
            pu = find(u)
            pv = find(v)
            if pu != pv:
                if size[pu] < size[pv]:
                    pu, pv = pv, pu
                parent[pv] = pu
                size[pu] += size[pv]
                merged = sorted(top[pu] + top[pv], reverse=True)[:10]
                top[pu] = merged
        else:
            v = int(input[index+1])
            k = int(input[index+2])
            index +=3
            pv = find(v)
            if len(top[pv]) >= k:
                output.append(str(top[pv][k-1]))
            else:
                output.append("-1")
    print('
'.join(output))
    
if __name__ == "__main__":
    main()