def v2(n):
    count = 0
    if n == 0:
        return 0
    while n > 0 and n % 2 == 0:
        n //= 2
        count += 1
    return count

def is_mountain_fold(crease_index):
    j = 100 - v2(crease_index)
    return j % 2 == 0

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    max_f = 0
    period = 8
    for i in range(1, period + 1):
        current_f = 0
        for k in range(n):
            crease_index = i + a[k]
            if crease_index >= 1 and crease_index <= (1 << 100) - 1:
                if is_mountain_fold(crease_index):
                    current_f += 1
        max_f = max(max_f, current_f)
    print(max_f)

if __name__ == '__main__':
    solve()