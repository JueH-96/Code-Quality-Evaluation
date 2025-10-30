import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    K = int(data[1])
    p = list(map(int, data[2:2+n]))
    p0 = [x-1 for x in p]
    
    visited = [False] * n
    res = [0] * n
    
    for i in range(n):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = p0[cur]
            L = len(cycle)
            if L > 0:
                exponent = pow(2, K, L)
            else:
                exponent = 0
            for j in range(L):
                new_index = (j + exponent) % L
                res[cycle[j]] = cycle[new_index]
                
    ans = [x+1 for x in res]
    print(" ".join(map(str, ans)))

if __name__ == '__main__':
    main()