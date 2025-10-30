# YOUR CODE HERE

import sys

def solve():
    n = int(sys.stdin.readline())
    cuboids = []
    for _ in range(n):
        cuboids.append(list(map(int, sys.stdin.readline().split())))

    for i in range(n):
        count = 0
        for j in range(n):
            if i != j:
                if cuboids[i][0] < cuboids[j][0] < cuboids[i][3] and cuboids[i][1] < cuboids[j][1] < cuboids[i][4] and cuboids[i][2] < cuboids[j][2] < cuboids[i][5]:
                    count += 1
                elif cuboids[i][0] < cuboids[j][3] < cuboids[i][3] and cuboids[i][1] < cuboids[j][4] < cuboids[i][4] and cuboids[i][2] < cuboids[j][5] < cuboids[i][5]:
                    count += 1
        print(count)

solve()