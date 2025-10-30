import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    L = int(input[idx]); idx +=1

    a = list(map(int, input[idx:idx+N]))
    idx += N
    b = list(map(int, input[idx:idx+M]))
    idx += M

    excluded = set()
    for _ in range(L):
        c = int(input[idx]); idx +=1
        d = int(input[idx]); idx +=1
        excluded.add((c, d))

    # Sort main dishes in descending order of a_i, keeping original indices (1-based)
    sorted_a = sorted([(a[i], i+1) for i in range(N)], key=lambda x: (-x[0], x[1]))
    # Sort side dishes similarly
    sorted_b = sorted([(b[i], i+1) for i in range(M)], key=lambda x: (-x[0], x[1]))

    heap = []
    visited = set()

    # Initialize with the first pair (0,0)
    initial_i = 0
    initial_j = 0
    initial_sum = sorted_a[initial_i][0] + sorted_b[initial_j][0]
    heapq.heappush(heap, (-initial_sum, initial_i, initial_j))
    visited.add((initial_i, initial_j))

    while heap:
        current = heapq.heappop(heap)
        current_sum = -current[0]
        i = current[1]
        j = current[2]

        main_idx = sorted_a[i][1]
        side_idx = sorted_b[j][1]

        if (main_idx, side_idx) not in excluded:
            print(current_sum)
            return

        # Explore next pairs
        for di, dj in [(1, 0), (0, 1)]:
            ni = i + di
            nj = j + dj
            if ni < len(sorted_a) and nj < len(sorted_b):
                if (ni, nj) not in visited:
                    visited.add((ni, nj))
                    new_sum = sorted_a[ni][0] + sorted_b[nj][0]
                    heapq.heappush(heap, (-new_sum, ni, nj))

    # The problem states at least one is available, so we should have returned

if __name__ == "__main__":
    main()