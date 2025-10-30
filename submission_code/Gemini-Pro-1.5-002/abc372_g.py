# YOUR CODE HERE
def solve():
    n = int(input())
    abc = []
    for _ in range(n):
        abc.append(list(map(int, input().split())))

    count = 0
    for x in range(1, 1001):
        for y in range(1, 1001):
            valid = True
            for i in range(n):
                if abc[i][0] * x + abc[i][1] * y >= abc[i][2]:
                    valid = False
                    break
            if valid:
                count += 1
    print(count)


t = int(input())
for _ in range(t):
    solve()