def solve():
    n = int(input())
    a = list(map(int, input().split()))

    middle_indices = {}
    for i in range(1, n + 1):
        indices = [index for index, val in enumerate(a) if val == i]
        middle_indices[i] = indices[1]

    sorted_numbers = sorted(range(1, n + 1), key=lambda x: middle_indices[x])
    print(*sorted_numbers)

solve()