def minimal_rotation(s):
    n = len(s)
    if n == 0:
        return s
    k = 0
    for j in range(1, n):
        i = k
        while i < j and s[i] < s[j]:
            i += 1
            j += 1
        if i == j:
            k = j
        else:
            if s[k] > s[j]:
                k = j
    return s[k:] + s[:k]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    P = list(map(int, data[idx:idx+N]))
    idx += N
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    visited = [False] * N
    ans = A.copy()
    
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j] - 1  # convert to 0-based index
            
            elements = [A[idx] for idx in cycle]
            rotated = minimal_rotation(elements)
            
            k = 0
            for pos in cycle:
                ans[pos] = rotated[k]
                k += 1
    
    print(' '.join(map(str, ans)))

if __name__ == '__main__':
    main()