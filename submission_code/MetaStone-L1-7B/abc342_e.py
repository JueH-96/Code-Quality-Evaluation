import sys
import heapq
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    train_dict = defaultdict(list)

    for _ in range(M):
        l = int(input[ptr])
        ptr += 1
        d = int(input[ptr])
        ptr += 1
        k = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        train_dict[B].append((l, d, k, c, A))

    f = [-float('inf')] * (N + 1)
    f[N] = 10**18

    pq = []

    for S in train_dict:
        if S == N:
            for train in train_dict[S]:
                l, d, k, c, A = train
                t_max = l + (k-1)*d
                if t_max > f[S]:
                    f[S] = t_max
                    heapq.heappush(pq, -S)

    while pq:
        current_S = -heapq.heappop(pq)
        if f[current_S] == -float('inf'):
            continue
        for train in train_dict[current_S]:
            l, d, k, c, T = train
            T_max = f[current_S] - c
            if T_max < l:
                continue
            max_possible = l + (k-1)*d
            if T_max > max_possible:
                T_max = max_possible
            if T_max < l:
                continue
            m_max = (T_max - l) // d
            if m_max < 0:
                continue
            m_max = min(m_max, k-1)
            if m_max < 0:
                continue
            t_candidate = l + m_max * d
            if t_candidate > f[T]:
                f[T] = t_candidate
                heapq.heappush(pq, -T)

    for S in range(1, N):
        if f[S] == -float('inf'):
            print('Unreachable')
        else:
            print(f[S])

if __name__ == '__main__':
    main()