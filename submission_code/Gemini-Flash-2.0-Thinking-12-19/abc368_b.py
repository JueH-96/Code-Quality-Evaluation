def solve():
    n = int(input())
    a = list(map(int, input().split()))
    operation_count = 0
    while True:
        positive_count = 0
        for x in a:
            if x > 0:
                positive_count += 1
        if positive_count <= 1:
            break
        a.sort(reverse=True)
        a[0] -= 1
        a[1] -= 1
        operation_count += 1
    print(operation_count)

if __name__ == '__main__':
    solve()