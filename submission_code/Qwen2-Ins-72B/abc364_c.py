import sys
from bisect import bisect_left

def main():
    input = sys.stdin.readline
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    AB = sorted([(a, b) for a, b in zip(A, B)], key=lambda x: x[0] + x[1])
    sweetness = 0
    saltiness = 0
    for i, (a, b) in enumerate(AB):
        sweetness += a
        saltiness += b
        AB[i] = (sweetness, saltiness)
    AB.append((X + 1, Y + 1))
    ans = bisect_left(AB, (X + 1, Y + 1))
    print(ans)

main()