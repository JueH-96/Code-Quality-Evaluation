import heapq

def main():
    import sys
    S = list(sys.stdin.readline().strip())
    n = len(S)
    heap = []
    
    # Initialize the heap with all starting indices of ABC in the original string
    for i in range(n - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            heapq.heappush(heap, i)
    
    while heap:
        i = heapq.heappop(heap)
        # Check if the current index is still valid and forms ABC
        if i + 2 < len(S) and S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            # Remove the ABC
            del S[i:i+3]
            # Check the surrounding area for new ABCs
            start = max(0, i - 2)
            end = min(len(S), i + 2)
            # Check all possible positions in the new string from start to end-2
            for j in range(start, end - 2 + 1):
                if j + 2 < len(S) and S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                    heapq.heappush(heap, j)
    
    print(''.join(S))

if __name__ == "__main__":
    main()