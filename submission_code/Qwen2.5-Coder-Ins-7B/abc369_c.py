# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

def count_arithmetic_progressions(A):
    count = 0
    for i in range(N):
        d = {}
        for j in range(i, N):
            if A[j] - A[i] in d:
                count += d[A[j] - A[i]]
            if A[j] not in d:
                d[A[j]] = 0
            d[A[j]] += 1
    return count

print(count_arithmetic_progressions(A))