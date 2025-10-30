def solve():
    n = int(input())
    a = list(map(int, input().split()))

    count = 0
    for j in range(1, n - 1):
        aj = a[j]
        for i in range(j):
            for k in range(j + 1, n):
                if a[i] == a[k] and a[i] != aj:
                    count += 1
    print(count)

if __name__ == "__main__":
    solve()