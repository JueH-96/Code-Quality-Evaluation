import sys

def main():
    data = sys.stdin
    line = data.readline().strip()
    if not line:
        return
    n = int(line)
    sum_a = 0
    max_diff = 0
    for _ in range(n):
        a, b = map(int, data.readline().split())
        sum_a += a
        diff = b - a
        if diff > max_diff:
            max_diff = diff
    # The best head height is sum of all A_i plus the maximum (B_i - A_i)
    print(sum_a + max_diff)

main()