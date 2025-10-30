import sys
import threading
def main():
    import sys
    import heapq

    input = sys.stdin.readline
    Q = int(input())
    # min-heap storing k_i = add_at_insert for each plant
    heap = []
    add = 0  # global added height offset
    out = []
    for _ in range(Q):
        parts = input().split()
        typ = int(parts[0])
        if typ == 1:
            # plant a new plant with base so that height = 0 => store k_i = add
            heapq.heappush(heap, add)
        elif typ == 2:
            # wait T days
            T = int(parts[1])
            add += T
        else:  # typ == 3
            H = int(parts[1])
            # we need to harvest all plants with effective height >= H
            # effective height of a plant inserted at time k_i is (add - k_i)
            # so add - k_i >= H  <=>  k_i <= add - H
            limit = add - H
            cnt = 0
            # pop all k_i <= limit
            while heap and heap[0] <= limit:
                heapq.heappop(heap)
                cnt += 1
            out.append(str(cnt))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()