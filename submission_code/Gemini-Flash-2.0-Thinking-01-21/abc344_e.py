def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        query = input().split()
        type = int(query[0])

        if type == 1:
            x = int(query[1])
            y = int(query[2])
            try:
                index_x = a.index(x)
                a.insert(index_x + 1, y)
            except ValueError:
                pass # x should exist, but handle for robustness

        elif type == 2:
            x = int(query[1])
            try:
                index_x = a.index(x)
                a.pop(index_x)
            except ValueError:
                pass # x should exist, but handle for robustness

    print(*(a))

if __name__ == "__main__":
    solve()