# YOUR CODE HERE
import sys
input = sys.stdin.readline
T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    if N == 2 * K:
        print("No")
    else:
        print("Yes")