import sys

def read_input():
    N, Q = map(int, input().split())
    K = [int(input()) for _ in range(Q)]
    return N, Q, K

def is_possible(N, K):
    if K == 0:
        return "Yes"
    if K == N**2:
        return "Yes"
    if K % 2 != 0:
        return "No"
    return "Yes"

def solve():
    N, Q, K = read_input()
    for k in K:
        print(is_possible(N, k))

if __name__ == "__main__":
    solve()