def solve():
    q = int(input())
    bag = {}
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x = query[1]
            if x not in bag:
                bag[x] = 0
            bag[x] += 1
        elif query[0] == 2:
            x = query[1]
            bag[x] -= 1
            if bag[x] == 0:
                del bag[x]
        else:
            print(len(bag))

solve()