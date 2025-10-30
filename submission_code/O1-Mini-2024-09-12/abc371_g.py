# YOUR CODE HERE
import sys

def min_cyclic_shift(s):
    n = len(s)
    s = s + s
    i, j, k = 0, 1, 0
    while i < n and j < n:
        k = 0
        while k < n and s[i + k] == s[j + k]:
            k += 1
        if k == n:
            break
        if s[i + k] > s[j + k]:
            i = i + k + 1
            if i == j:
                i +=1
        else:
            j = j + k +1
            if j == i:
                j +=1
    pos = min(i, j)
    return pos

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    P = list(map(int, data[idx:idx+N])); idx +=N
    A = list(map(int, data[idx:idx+N])); idx +=N
    P = [p-1 for p in P]  # 0-based
    visited = [False]*N
    res = [0]*N
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j]
            if len(cycle) ==1:
                res[cycle[0]] = A[cycle[0]]
            else:
                a_vals = [A[pos] for pos in cycle]
                shift = min_cyclic_shift(a_vals)
                shifted = a_vals[shift:] + a_vals[:shift]
                for pos, val in zip(cycle, shifted):
                    res[pos] = val
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()