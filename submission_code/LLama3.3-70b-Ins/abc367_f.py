from collections import Counter

def solve():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    for _ in range(Q):
        l, r, L, R = map(int, input().split())
        sub_A = A[l-1:r]
        sub_B = B[L-1:R]

        if Counter(sub_A) == Counter(sub_B):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()