import heapq
import sys

def main():
    n = int(sys.stdin.readline())
    chords = []
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        if a > b:
            a, b = b, a
        chords.append((a, b))
    
    chords.sort()
    heap = []
    
    for a, b in chords:
        while heap and heap[0] <= a:
            heapq.heappop(heap)
        if heap and b > heap[0]:
            print("Yes")
            return
        heapq.heappush(heap, b)
    
    print("No")

if __name__ == "__main__":
    main()