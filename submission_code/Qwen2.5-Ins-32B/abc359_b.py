import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    count = 0
    for i in range(1, N+1):
        indices = [j for j, x in enumerate(A) if x == i]
        if abs(indices[0] - indices[1]) == 2:
            count += 1
    print(count)

if __name__ == "__main__":
    solve()