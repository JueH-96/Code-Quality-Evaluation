def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = input()

    def calculate_mex(arr):
        seen = set(arr)
        i = 0
        while i in seen:
            i += 1
        return i

    total_mex_sum = 0
    for i in range(n):
        if s[i] == 'M':
            for j in range(i + 1, n):
                if s[j] == 'E':
                    for k in range(j + 1, n):
                        if s[k] == 'X':
                            total_mex_sum += calculate_mex([a[i], a[j], a[k]])
    print(total_mex_sum)

solve()