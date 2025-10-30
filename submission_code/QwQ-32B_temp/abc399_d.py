import sys
from collections import defaultdict, Counter

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+2*N]))
        ptr += 2*N

        # Store positions for each number (1-based)
        pos = [[] for _ in range(N+1)]
        for idx in range(2*N):
            num = A[idx]
            pos[num].append(idx + 1)  # positions are 1-based

        is_adjacent = [False] * (N+1)
        for a in range(1, N+1):
            p1, p2 = pos[a]
            if abs(p1 - p2) == 1:
                is_adjacent[a] = True

        # Compute consecutive four case
        count_consec = defaultdict(int)
        max_i = 2 * N - 3
        for i in range(1, 2*N - 2):  # i is 1-based position's first element of the four
            # The four positions are i, i+1, i+2, i+3 (1-based)
            # The indices in A (0-based) are i-1, i, i+1, i+2
            elements = [A[i-1], A[i], A[i+1], A[i+2]]
            freq = Counter(elements)
            if len(freq) != 2:
                continue
            a, b = sorted(freq.keys())
            if a == b:
                continue
            if not is_adjacent[a] and not is_adjacent[b]:
                count_consec[(a, b)] += 1

        # Compute separated pairs case
        count_separated = defaultdict(int)
        prev_counts = defaultdict(int)
        for i in range(1, 2*N):  # i is the first element of the adjacent pair (1-based)
            x = A[i-1]
            y = A[i]
            sorted_pair = tuple(sorted((x, y)))
            # Add the count from previous occurrences
            count_separated[sorted_pair] += prev_counts[sorted_pair]
            # Update prev_counts for future pairs
            prev_counts[sorted_pair] += 1

        total = 0

        # Process separated pairs
        for (a, b), cnt in count_separated.items():
            if a == b:
                continue
            if not is_adjacent[a] and not is_adjacent[b]:
                total += cnt

        # Process consecutive case
        for (a, b), cnt in count_consec.items():
            total += cnt  # already checked conditions in count_consec

        print(total)

if __name__ == "__main__":
    main()