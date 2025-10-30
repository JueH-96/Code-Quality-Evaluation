# YOUR CODE HERE
import sys

def max_subsidy_limit(N, M, A):
    if sum(A) <= M:
        print("infinite")
        return

    A.sort()
    low, high = 0, 10**9
    while low < high:
        mid = (low + high + 1) // 2
        total_subsidy = sum(min(mid, a) for a in A)
        if total_subsidy <= M:
            low = mid
        else:
            high = mid - 1

    print(low)

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

max_subsidy_limit(N, M, A)