import heapq
import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); m = int(data[index+1]); index += 2
        boxes = []
        for i in range(n):
            v = int(data[index]); p = int(data[index+1]); index += 2
            boxes.append((p, v))
        
        boxes.sort(key=lambda x: x[0])
        heap = []
        ans = 0
        count = 0
        for (p, v) in boxes:
            if count < m:
                count += 1
                ans += v - p
                heapq.heappush(heap, -v)
            else:
                if heap and -heap[0] > v:
                    largest = -heapq.heappop(heap)
                    ans = ans - largest + v
                    heapq.heappush(heap, -v)
        if ans < 0:
            ans = 0
        results.append(str(ans))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()