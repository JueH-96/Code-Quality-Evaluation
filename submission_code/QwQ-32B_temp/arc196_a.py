import sys
import heapq

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    if N == 0:
        print(0)
        return
    prev = [i - 1 for i in range(N)]
    prev[0] = -1
    next_ = [i + 1 for i in range(N)]
    next_[-1] = -1
    heap = []
    for i in range(N - 1):
        diff = abs(A[i] - A[i + 1])
        heapq.heappush(heap, (-diff, i, i + 1))
    total = 0
    count = N
    while count > 1:
        while True:
            if not heap:
                break
            current = heapq.heappop(heap)
            current_diff = -current[0]
            left = current[1]
            right = current[2]
            if next_[left] == right and prev[right] == left:
                break
        total += current_diff
        prev_node = prev[left]
        next_node = next_[right]
        if prev_node != -1:
            next_[prev_node] = next_node
        if next_node != -1:
            prev[next_node] = prev_node
        if prev_node != -1 and next_node != -1:
            new_diff = abs(A[prev_node] - A[next_node])
            heapq.heappush(heap, (-new_diff, prev_node, next_node))
        count -= 2
    print(total)

if __name__ == "__main__":
    main()