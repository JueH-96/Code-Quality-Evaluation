def solve():
    n = int(input())
    p = list(map(int, input().split()))
    if n == 1:
        print(0)
    else:
        max_other_score = 0
        for i in range(1, n):
            max_other_score = max(max_other_score, p[i])
        needed_increase = 0
        if p[0] <= max_other_score:
            needed_increase = max_other_score - p[0] + 1
        else:
            needed_increase = 0
        print(needed_increase)

if __name__ == '__main__':
    solve()