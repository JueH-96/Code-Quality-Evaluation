import sys
import threading
import heapq

def main():
    S = list(sys.stdin.readline().rstrip())
    n = len(S)
    # Min-heap of starting indices of "WA"
    heap = []
    for i in range(n - 1):
        if S[i] == 'W' and S[i+1] == 'A':
            heapq.heappush(heap, i)

    # Repeatedly replace the leftmost "WA" with "AC"
    while heap:
        i = heapq.heappop(heap)
        # Check if this occurrence is still valid
        if S[i] != 'W' or i+1 >= n or S[i+1] != 'A':
            continue
        # Perform the replacement
        S[i], S[i+1] = 'A', 'C'
        # After replacement, only a new "WA" could appear starting at i-1
        j = i - 1
        if j >= 0 and S[j] == 'W' and S[j+1] == 'A':
            heapq.heappush(heap, j)

    # Output the final string
    sys.stdout.write(''.join(S))

if __name__ == "__main__":
    main()