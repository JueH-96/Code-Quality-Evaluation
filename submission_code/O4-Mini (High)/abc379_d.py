import sys
import heapq

def main():
    input = sys.stdin.readline
    Q = int(input())
    global_add = 0
    heap = []
    answers = []

    for _ in range(Q):
        data = input().split()
        t = data[0]
        if t == '1':
            # Plant a new seedling: record the global offset at creation
            heapq.heappush(heap, global_add)
        elif t == '2':
            # Wait T days: increase the global offset
            T = int(data[1])
            global_add += T
        else:  # t == '3'
            # Harvest all plants with height >= H
            H = int(data[1])
            # A plant created when global_add was g_i has current height = global_add - g_i.
            # We need global_add - g_i >= H  =>  g_i <= global_add - H
            threshold = global_add - H
            cnt = 0
            # Pop all creation-times g_i <= threshold
            while heap and heap[0] <= threshold:
                heapq.heappop(heap)
                cnt += 1
            answers.append(str(cnt))

    # Output all harvest counts
    sys.stdout.write("
".join(answers))

if __name__ == "__main__":
    main()