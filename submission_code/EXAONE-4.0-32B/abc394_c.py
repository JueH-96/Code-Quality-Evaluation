import heapq
import sys

def main():
    data = sys.stdin.readline().strip()
    if not data:
        return
    arr = list(data)
    n = len(arr)
    heap = []
    
    for i in range(n-1):
        if arr[i] == 'W' and arr[i+1] == 'A':
            heapq.heappush(heap, i)
            
    while heap:
        i = heapq.heappop(heap)
        if i < 0 or i >= len(arr) - 1:
            continue
        if arr[i] != 'W' or arr[i+1] != 'A':
            continue
        arr[i] = 'A'
        arr[i+1] = 'C'
        if i-1 >= 0 and arr[i-1] == 'W' and arr[i] == 'A':
            heapq.heappush(heap, i-1)
            
    print(''.join(arr))

if __name__ == "__main__":
    main()