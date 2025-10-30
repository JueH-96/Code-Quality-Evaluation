import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1

    H = [0] + [int(data[i]) for i in range(index, index + N)]
    index += N

    queries = []
    for _ in range(Q):
        l = int(data[index])
        index += 1
        r = int(data[index])
        index += 1
        queries.append((l, r))

    # Precompute the next higher building index for each building
    next_higher = [0] * (N + 1)
    stack = []

    for i in range(1, N + 1):
        while stack and H[stack[-1]] < H[i]:
            next_higher[stack.pop()] = i
        stack.append(i)

    # Precompute the previous higher building index for each building
    prev_higher = [0] * (N + 1)
    stack = []

    for i in range(1, N + 1):
        while stack and H[stack[-1]] < H[i]:
            prev_higher[stack.pop()] = i
        stack.append(i)

    results = []

    for l, r in queries:
        count = 0
        current = next_higher[r]

        while current <= N:
            if prev_higher[current] <= l:
                count += 1
            current = next_higher[current]

        results.append(count)

    for result in results:
        print(result)

solve()