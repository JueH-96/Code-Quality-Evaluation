# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    flavors = []
    deliciousness = []
    index = 1
    for i in range(N):
        flavors.append(int(input[index]))
        deliciousness.append(int(input[index + 1]))
        index += 2

    max_heap = {}
    for i in range(N):
        if flavors[i] not in max_heap:
            max_heap[flavors[i]] = []
        heapq.heappush(max_heap[flavors[i]], -deliciousness[i])

    max_two = []
    for flavor in max_heap:
        if len(max_heap[flavor]) >= 2:
            first = -heapq.heappop(max_heap[flavor])
            second = -heapq.heappop(max_heap[flavor])
            heapq.heappush(max_heap[flavor], -second)
            max_two.append(first + second // 2)
        elif len(max_heap[flavor]) == 1:
            first = -heapq.heappop(max_heap[flavor])
            max_two.append(first)

    max_two.sort(reverse=True)
    if len(max_two) >= 2:
        max_satisfaction = max(max_two[0], max_two[1])
    else:
        max_satisfaction = max_two[0]

    for flavor in max_heap:
        if len(max_heap[flavor]) == 1:
            first = -heapq.heappop(max_heap[flavor])
            for other_flavor in max_heap:
                if other_flavor != flavor and max_heap[other_flavor]:
                    second = -heapq.heappop(max_heap[other_flavor])
                    max_satisfaction = max(max_satisfaction, first + second)
                    heapq.heappush(max_heap[other_flavor], -second)

    print(max_satisfaction)

if __name__ == "__main__":
    main()