def solve():
    n = int(input())
    a = list(map(int, input().split()))

    while True:
        modified = False
        i = 0
        while i < len(a) - 1:
            if abs(a[i] - a[i+1]) != 1:
                insertion = []
                if a[i] < a[i+1]:
                    insertion = list(range(a[i] + 1, a[i+1]))
                else:
                    insertion = list(range(a[i] - 1, a[i+1], -1))

                a = a[:i+1] + insertion + a[i+1:]
                modified = True
                break
            i += 1

        if not modified:
            break

    print(*a)

# YOUR CODE HERE
solve()