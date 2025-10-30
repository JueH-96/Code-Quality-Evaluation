import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    P = list(map(int, data[2:2+n]))
    p = [x-1 for x in P]

    visited = [False] * n
    cycles = []

    for i in range(n):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = p[cur]
            cycles.append(cycle)
    
    res_arr = [0] * n
    for cycle in cycles:
        L = len(cycle)
        shift = pow(2, k, L)
        for j in range(L):
            idx = cycle[j]
            target_index = (j + shift) % L
            res_arr[idx] = cycle[target_index]
            
    res_arr = [x + 1 for x in res_arr]
    print(" ".join(map(str, res_arr)))

if __name__ == "__main__":
    main()