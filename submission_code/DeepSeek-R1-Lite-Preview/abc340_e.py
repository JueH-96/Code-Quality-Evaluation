def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    current = A.copy()
    diff = [0] * (N + 1)
    from collections import defaultdict
    ops = defaultdict(list)
    
    for i in range(M):
        ops[B[i]].append(i)
    
    for j in range(N):
        if j in ops:
            for _ in ops[j]:
                K = current[j]
                current[j] -= K
                s = (j + 1) % N
                e = (s + K - 1) % N
                if s <= e:
                    diff[s] += 1
                    if e + 1 < N:
                        diff[e + 1] -= 1
                else:
                    diff[s] += 1
                    if e + 1 < N:
                        diff[e + 1] -= 1
                    diff[0] += 1
    
    additions = [0] * N
    additions[0] = diff[0]
    for j in range(1, N):
        additions[j] = additions[j-1] + diff[j]
    
    X = [0] * N
    for j in range(N):
        X[j] = current[j] + additions[j]
    
    print(' '.join(map(str, X)))

if __name__ == '__main__':
    main()