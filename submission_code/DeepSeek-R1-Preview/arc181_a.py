import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        P = list(map(int, data[idx:idx+N]))
        idx += N
        
        # Check if already sorted
        sorted_flag = True
        for i in range(N):
            if P[i] != i + 1:
                sorted_flag = False
                break
        if sorted_flag:
            print(0)
            continue
        
        # Compute prefix_max, suffix_min, suffix_max
        prefix_max = [0] * N
        prefix_max[0] = P[0]
        for i in range(1, N):
            prefix_max[i] = max(prefix_max[i-1], P[i])
        
        suffix_min = [0] * N
        suffix_max = [0] * N
        suffix_min[-1] = P[-1]
        suffix_max[-1] = P[-1]
        for i in range(N-2, -1, -1):
            suffix_min[i] = min(P[i], suffix_min[i+1])
            suffix_max[i] = max(P[i], suffix_max[i+1])
        
        # Check for each k
        found = False
        for k in range(1, N+1):
            pos = k - 1
            if P[pos] != k:
                continue
            if k == 1:
                if N >= 2 and suffix_min[1] == 2 and suffix_max[1] == N:
                    found = True
                    break
            elif k == N:
                if prefix_max[N-2] == N - 1:
                    found = True
                    break
            else:
                if prefix_max[k-2] != k - 1:
                    continue
                if suffix_min[k] == k + 1 and suffix_max[k] == N:
                    found = True
                    break
        
        if found:
            print(1)
        else:
            print(2)

if __name__ == '__main__':
    main()