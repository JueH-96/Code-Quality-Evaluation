# YOUR CODE HERE
def count_valid_pairs(N, A, B, C):
    def binary_search(f, lo, hi):
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return lo

    max_x = min((c - 1) // a for a, c in zip(A, C) if a > 0)
    count = 0

    for x in range(1, max_x + 1):
        min_y = 1
        max_y = float('inf')

        for a, b, c in zip(A, B, C):
            if b == 0:
                if a * x >= c:
                    max_y = 0
                    break
            else:
                y = (c - a * x - 1) // b
                if y < min_y:
                    min_y = y + 1
                if y < max_y:
                    max_y = y

        if min_y <= max_y:
            count += max_y - min_y + 1

    return count

T = int(input())
for _ in range(T):
    N = int(input())
    A, B, C = [], [], []
    for _ in range(N):
        a, b, c = map(int, input().split())
        A.append(a)
        B.append(b)
        C.append(c)
    print(count_valid_pairs(N, A, B, C))