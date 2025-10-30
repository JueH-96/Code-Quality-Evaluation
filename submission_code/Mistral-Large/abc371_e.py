import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store the last occurrence of each number
    last_occurrence = {}
    # Array to store the count of distinct elements up to each index
    distinct_count = [0] * (N + 1)

    total_sum = 0

    for i in range(N):
        if A[i] in last_occurrence:
            last_index = last_occurrence[A[i]]
            total_sum += (i - last_index) * (distinct_count[last_index] + 1)
            total_sum -= (i - last_index) * (distinct_count[last_index])
        else:
            total_sum += i + 1

        distinct_count[i + 1] = distinct_count[i]
        if i == 0 or A[i] != A[i - 1]:
            distinct_count[i + 1] += 1

        last_occurrence[A[i]] = i

    print(total_sum)

if __name__ == "__main__":
    solve()