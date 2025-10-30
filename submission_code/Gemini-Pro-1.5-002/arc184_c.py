def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def get_fold_type(x):
        binary_x = bin(x + 1)[2:]
        count = 0
        for bit in binary_x:
            if bit == '1':
                count += 1
        return "M" if count % 2 == 1 else "V"

    max_mountain_folds = 0
    for i in range(min(1000, 2**20 - a[-1] -1 )):
        mountain_folds = 0
        for j in range(n):
            if get_fold_type(i + a[j]) == "M":
                mountain_folds += 1
        max_mountain_folds = max(max_mountain_folds, mountain_folds)
    
    print(max_mountain_folds)

solve()