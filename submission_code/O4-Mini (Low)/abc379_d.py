import sys
import threading

def main():
    import sys
    import heapq
    
    input = sys.stdin.readline
    Q = int(input())
    heap = []
    offset = 0
    out_lines = []
    
    for _ in range(Q):
        line = input().split()
        typ = int(line[0])
        if typ == 1:
            # Plant a new plant of height 0 => base representation = -offset
            # We store val = offset_at_plant = offset
            heapq.heappush(heap, offset)
        elif typ == 2:
            # Increase all by T
            T = int(line[1])
            offset += T
        else:
            # Harvest all plants with actual height >= H
            H = int(line[1])
            # actual height = offset - val  >= H  =>  val <= offset - H
            thresh = offset - H
            cnt = 0
            # heap is min-heap of val = offset_at_plant
            while heap and heap[0] <= thresh:
                heapq.heappop(heap)
                cnt += 1
            out_lines.append(str(cnt))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()