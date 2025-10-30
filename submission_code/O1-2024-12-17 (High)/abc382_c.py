def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))

    # Pair each person's threshold with their (1-based) index and sort by threshold
    people = [(A[i], i+1) for i in range(N)]
    people.sort(key=lambda x: x[0])  # sort by threshold ascending

    # Pair each sushi's deliciousness with its index and sort by deliciousness
    sushis = [(B[j], j) for j in range(M)]
    sushis.sort(key=lambda x: x[0])  # sort by deliciousness ascending

    import heapq
    heap = []  # will store the indices (person numbers) of those who can eat so far
    p = 0      # pointer to move through "people"
    ans = [-1]*M

    # For each sushi in ascending order of deliciousness:
    for deliciousness, sushi_idx in sushis:
        # Include all people whose threshold is <= this sushi's deliciousness
        while p < N and people[p][0] <= deliciousness:
            # Push the person's index onto the min-heap
            # so the smallest (earliest) index is always on top
            heapq.heappush(heap, people[p][1])
            p += 1

        # If we have at least one person who can eat this sushi,
        # the earliest such person is the top of the heap
        if heap:
            ans[sushi_idx] = heap[0]
        else:
            ans[sushi_idx] = -1

    # Print the answers in the original sushi order
    print('
'.join(map(str, ans)))


# Don't forget to call main()!
if __name__ == "__main__":
    main()