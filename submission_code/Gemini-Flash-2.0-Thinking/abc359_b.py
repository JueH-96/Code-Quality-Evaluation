def solve():
    n = int(input())
    a = list(map(int, input().split()))

    count = 0
    for color in range(1, n + 1):
        indices = [i for i, x in enumerate(a) if x == color]
        if len(indices) == 2:
            index1, index2 = indices[0], indices[1]
            if index2 - index1 - 1 == 1:
                count += 1
    print(count)

# YOUR CODE HERE
solve()