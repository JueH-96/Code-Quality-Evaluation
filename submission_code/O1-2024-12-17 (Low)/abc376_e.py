def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1

    results = []
    for _ in range(t):
        N = int(input_data[idx]); idx += 1
        K = int(input_data[idx]); idx += 1

        A = list(map(int, input_data[idx:idx+N]))
        idx += N
        B = list(map(int, input_data[idx:idx+N]))
        idx += N

        pairs = [(A[i], B[i]) for i in range(N)]
        # sort by A ascending
        pairs.sort(key=lambda x: x[0])

        # max-heap for B (store negative B to simulate max-heap using heapq)
        heap = []
        b_sum = 0
        best = float('inf')

        for a_val, b_val in pairs:
            # add current b_val
            heapq.heappush(heap, -b_val)
            b_sum += b_val

            # if we exceed K elements, remove the largest B
            if len(heap) > K:
                largest_b = -heapq.heappop(heap)
                b_sum -= largest_b

            # if we have exactly K elements, check the product
            if len(heap) == K:
                candidate = a_val * b_sum
                if candidate < best:
                    best = candidate

        results.append(str(best))

    print("
".join(results))

# Do not forget to call main!
if __name__ == "__main__":
    main()