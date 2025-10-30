from bisect import bisect_left, bisect_right
N = int(input())
p = [int(x) for x in input().split()]
M = int(input())
A = [int(x) for x in input().split()]

def BIT_add(bit, i):
    while i <= N:
        bit[i] += 1
        i += i & (-i)

def BIT_sum(bit, i):
    result = 0
    while i > 0:
        result += bit[i]
        i -= i & (-i)
    return result

# make bit which is used as a counter of data in each segment
bit = [0] * (N + 1)
for i in range(N):
    BIT_add(bit, p[i])

now_inversion = 0
results = []
points = [N] * N
for i in p:
    idx = bisect_left(points, i)
    points[idx] = i-1
    now_inversion += idx * (bit[i] - 1) + i - 1 - (N - 1 - i) * idx - BIT_sum(bit, i) * idx
results.append(now_inversion)

# Solve while Parse
for i in range(M):
    a = A[i]

    # propagate new data that seems to be added by this operation
    for i in range(N - 1, a - 2, -1):
        if points[i] < points[i - 1]:
            l0, l1 = i - 1, i - 1
            r0, r1 = points[i - 1], points[i]
            d0 = 1
            d1 = 0
        else:
            l0, l1 = i, i
            r0, r1 = points[i], points[i - 1]
            d0 = 1
            d1 = -1

        # l0, r0, d0: Shifted segment as in [l0, r0), add d0.
        # l1, r1, d1: Shifted segment as in [l1, r1), add d1.
        if i != N-1: points[i + 1] = points[i]
        points[i] = points[i - 1]

        # Update bit and results
        inv = now_inversion
        while r0 - l0 and d0:
            while r0 <= N and (bit[r0]):
                if r0 + d0 <= N or r0 + d0 > l0:
                    BIT_add(bit, r0 + d0)
                    BIT_add(bit, r0, -1)
                    inv += (N - r0) * d0 - l0 * d0
                r0 += d0
            while l0 >= 1 and (bit[l0]):
                if l0 + d0 > 0 or l0 + d0 < r0:
                    BIT_add(bit, l0 + d0)
                    BIT_add(bit, l0, -1)
                    inv += - (N - l0) * d0 + r0 * d0
                l0 += d0
        while r1 - l1 and d1:
            while r1 <= N and (bit[r1]):
                if r1 + d1 <= N or r1 + d1 > l1:
                    BIT_add(bit, r1 + d1)
                    BIT_add(bit, r1, -1)
                    inv += (N - r1) * d1 - l1 * d1
                r1 += d1
            while l1 >= 1 and (bit[l1]):
                if l1 + d1 > 0 or l1 + d1 < r1:
                    BIT_add(bit, l1 + d1)
                    BIT_add(bit, l1, -1)
                    inv += - (N - l1) * d1 + r1 * d1
                l1 += d1

    points[a-1] = N
    if a != A[i-1]:
        results.append(inv)
        now_inversion = inv

for i in results:
    print(i)