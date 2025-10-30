import sys
from collections import Counter

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1

    A = list(map(int, data[index:index + N]))
    index += N
    B = list(map(int, data[index:index + N]))
    index += N

    queries = []
    for _ in range(Q):
        l, r, L, R = map(int, data[index:index + 4])
        index += 4
        queries.append((l, r, L, R))

    for l, r, L, R in queries:
        sub_A = A[l-1:r]
        sub_B = B[L-1:R]

        if Counter(sub_A) == Counter(sub_B):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()