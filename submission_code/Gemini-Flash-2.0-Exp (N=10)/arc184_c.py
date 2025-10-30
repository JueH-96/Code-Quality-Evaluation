def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def get_fold_type(index):
        binary = bin(index + 1)[2:].zfill(100)
        count = 0
        for bit in binary:
            if bit == '1':
                count += 1
        if count % 2 == 1:
            return 'M'
        else:
            return 'V'

    max_mountain_count = 0
    for i in range(1, (1 << 100) - a[-1]):
        mountain_count = 0
        for k in range(n):
            if get_fold_type(i + a[k] - 1) == 'M':
                mountain_count += 1
        max_mountain_count = max(max_mountain_count, mountain_count)
    
    print(max_mountain_count)

solve()