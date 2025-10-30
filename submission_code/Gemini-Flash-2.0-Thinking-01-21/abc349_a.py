def solve():
    n = int(input())
    a_values = list(map(int, input().split()))
    sum_a = sum(a_values)
    result = -sum_a
    print(result)

if __name__ == '__main__':
    solve()