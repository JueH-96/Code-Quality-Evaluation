import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    for i in range(M):
        bi = B[i]
        balls_in_hand = A[bi]
        A[bi] = 0
        c = 0
        while balls_in_hand > 0:
            next_box = (bi + c) % N
            A[next_box] += 1
            balls_in_hand -= 1
            c += 1

    print(*A)

if __name__ == "__main__":
    solve()