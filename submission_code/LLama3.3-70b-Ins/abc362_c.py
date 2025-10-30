import sys

def solve():
    N = int(input())
    intervals = []
    for _ in range(N):
        L, R = map(int, input().split())
        intervals.append((L, R))

    total_sum = sum((L + R) // 2 for L, R in intervals)
    if total_sum != 0:
        print("No")
        return

    X = []
    for L, R in intervals:
        if L % 2 == 0 and R % 2 == 0:
            X.append(L // 2)
        elif L % 2 == 1 and R % 2 == 1:
            X.append((L + 1) // 2)
        else:
            X.append(L // 2)

    print("Yes")
    print(*X)

if __name__ == "__main__":
    solve()