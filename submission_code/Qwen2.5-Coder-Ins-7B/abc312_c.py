# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:]))

A.sort()
B.sort()

left, right = 0, 10**9
while left < right:
    mid = (left + right) // 2
    sellers = len([a for a in A if a <= mid])
    buyers = len([b for b in B if b >= mid])
    if sellers >= buyers:
        right = mid
    else:
        left = mid + 1

print(left)