import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    tasks = []
    for _ in range(Q):
        T = int(input[ptr])
        G = int(input[ptr+1])
        tasks.append((T, G))
        ptr += 2
    
    total_steps = 0
    for T, G in tasks:
        idx = T - 1
        old = X[idx]
        delta = abs(G - old)
        total_steps += delta
        X[idx] = G
        q = deque()
        q.append(idx - 1)
        q.append(idx + 1)
        while q:
            j = q.popleft()
            if j < 0 or j >= N:
                continue
            min_valid = -float('inf')
            if j > 0:
                min_valid = X[j-1] + 1
            max_valid = float('inf')
            if j < N - 1:
                max_valid = X[j+1] - 1
            current = X[j]
            new_pos = current
            adjust = False
            if current < min_valid:
                new_pos = min_valid
                adjust = True
            elif current > max_valid:
                new_pos = max_valid
                adjust = True
            if adjust:
                total_steps += abs(new_pos - current)
                X[j] = new_pos
                q.append(j - 1)
                q.append(j + 1)
    print(total_steps)

if __name__ == '__main__':
    main()