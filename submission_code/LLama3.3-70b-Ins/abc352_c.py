import sys

def solve():
    N = int(input())
    giants = []
    for _ in range(N):
        A, B = map(int, input().split())
        giants.append((A, B))

    giants.sort(key=lambda x: x[1] - x[0], reverse=True)

    max_height = 0
    for i in range(N):
        height = 0
        for j in range(i, N):
            height += giants[j][0]
            max_height = max(max_height, height + giants[j][1])

    print(max_height)

if __name__ == "__main__":
    solve()