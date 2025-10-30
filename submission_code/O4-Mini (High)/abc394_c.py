import sys
import heapq

def main():
    S = sys.stdin.readline().strip()
    n = len(S)
    # Convert to list for mutable updates
    s = list(S)

    # Build a min‐heap of all starting indices of "WA"
    heap = []
    for i in range(n - 1):
        if s[i] == 'W' and s[i+1] == 'A':
            heap.append(i)
    heapq.heapify(heap)

    # Repeatedly replace the leftmost "WA" → "AC"
    # We always pop the smallest index i; if it's still "WA", do the replacement.
    # Only a new "WA" can appear at i-1, so we check and re‐push that.
    while heap:
        i = heapq.heappop(heap)
        # confirm it's still "WA"
        if s[i] == 'W' and s[i+1] == 'A':
            # perform replacement
            s[i]   = 'A'
            s[i+1] = 'C'
            # only index i-1 could form a new "WA"
            j = i - 1
            if j >= 0 and s[j] == 'W' and s[j+1] == 'A':
                heapq.heappush(heap, j)

    # Output result
    sys.stdout.write(''.join(s))


if __name__ == "__main__":
    main()