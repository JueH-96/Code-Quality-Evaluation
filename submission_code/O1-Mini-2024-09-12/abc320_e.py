# YOUR CODE HERE
def main():
    import sys, heapq
    from sys import stdout
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    events = []
    for _ in range(M):
        T = int(input[idx]); idx +=1
        W = int(input[idx]); idx +=1
        S = int(input[idx]); idx +=1
        events.append((T, W, S))
    # Initialize noodles
    noodles = [0] * N
    # Initialize segment tree
    size = 1
    while size < N:
        size <<=1
    INF = N+1
    tree = [INF] * (2 * size)
    for i in range(N):
        tree[size + i] = i +1
    for i in range(size -1, 0, -1):
        tree[i] = min(tree[2*i], tree[2*i +1])
    # Initialize return_heap
    return_heap = []
    heapq.heapify(return_heap)
    for event in events:
        T, W, S = event
        # Process all returns up to time T
        while return_heap and return_heap[0][0] <= T:
            return_time, person = heapq.heappop(return_heap)
            pos = person -1
            tree[size + pos] = person
            i = (size + pos) //2
            while i >=1:
                new_val = min(tree[2*i], tree[2*i +1])
                if tree[i] == new_val:
                    break
                tree[i] = new_val
                i = i //2
        # Get front person
        front = tree[1]
        if front <= N:
            noodles[front -1] += W
            # Remove front person
            pos = front -1
            tree[size + pos] = INF
            i = (size + pos) //2
            while i >=1:
                new_val = min(tree[2*i], tree[2*i +1])
                if tree[i] == new_val:
                    break
                tree[i] = new_val
                i = i //2
            # Schedule return
            heapq.heappush(return_heap, (T + S, front))
    # Output
    write = stdout.write
    for val in noodles:
        write(str(val) + '
')
if __name__ == "__main__":
    main()