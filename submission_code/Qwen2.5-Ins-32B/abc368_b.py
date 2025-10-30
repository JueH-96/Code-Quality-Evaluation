import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    count = 0
    while len([x for x in A if x > 0]) > 1:
        A.sort(reverse=True)
        A[0] -= 1
        A[1] -= 1
        count += 1
    print(count)

if __name__ == "__main__":
    solve()