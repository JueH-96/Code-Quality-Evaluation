import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(input[ptr]); ptr += 1
        b = int(input[ptr]); ptr += 1
        adj[a].append(b)
        adj[b].append(a)

    max_stamina = [-1] * (N + 1)
    heap = []

    for _ in range(K):
        p = int(input[ptr]); ptr += 1
        h = int(input[ptr]); ptr += 1
        max_stamina[p] = h
        heapq.heappush(heap, (-h, p))

    while heap:
        current_neg, u = heapq.heappop(heap)
        current = -current_neg
        if current < max_stamina[u]:
            continue
        for v in adj[u]:
            new_stamina = current - 1
            if new_stamina > max_stamina[v]:
                max_stamina[v] = new_stamina
                heapq.heappush(heap, (-new_stamina, v))

    result = [i for i in range(1, N + 1) if max_stamina[i] >= 0]
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()