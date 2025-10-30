import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    counts = {}
    heap = []
    index = 1
    for _ in range(n):
        s = int(data[index])
        c = int(data[index + 1])
        counts[s] = c
        heapq.heappush(heap, s)
        index += 2
    
    total = 0
    while heap:
        s = heapq.heappop(heap)
        if s not in counts:
            continue
        c = counts.pop(s)
        remainder = c % 2
        total += remainder
        carry = c // 2
        if carry > 0:
            new_s = s * 2
            if new_s in counts:
                counts[new_s] += carry
            else:
                counts[new_s] = carry
            heapq.heappush(heap, new_s)
    print(total)

if __name__ == "__main__":
    main()