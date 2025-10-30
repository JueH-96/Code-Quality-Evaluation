import sys

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    current_sum = 0
    min_prefix_sum = 0
    for i in range(n):
        current_sum += a[i]
        min_prefix_sum = min(min_prefix_sum, current_sum)
    result = max(0, -min_prefix_sum) + current_sum
    print(result)

if __name__ == '__main__':
    solve()