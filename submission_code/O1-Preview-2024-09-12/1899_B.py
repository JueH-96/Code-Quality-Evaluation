# YOUR CODE HERE
import sys
import threading
def main():
    import math
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a = list(map(int, sys.stdin.readline().split()))
        s = [0] * (n + 1)  # prefix sums
        for i in range(n):
            s[i + 1] = s[i] + a[i]
        divisors = set()
        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        max_diff = 0
        for k in divisors:
            group_sums = []
            if n % k != 0:
                continue
            valid = True
            for i in range(0, n, k):
                group_sums.append(s[i + k] - s[i])
            if len(group_sums) == 1:
                diff = 0
            else:
                max_sum = max(group_sums)
                min_sum = min(group_sums)
                diff = max_sum - min_sum
            if diff > max_diff:
                max_diff = diff
        print(max_diff)
threading.Thread(target=main).start()