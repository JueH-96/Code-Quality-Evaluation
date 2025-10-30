def solve():
    n = int(input())
    p = list(map(int, input().split()))
    if n == 1:
        print(0)
    else:
        max_other_ability = 0
        for i in range(1, n):
            max_other_ability = max(max_other_ability, p[i])
        if p[0] > max_other_ability:
            print(0)
        else:
            print(max_other_ability - p[0] + 1)

if __name__ == '__main__':
    solve()