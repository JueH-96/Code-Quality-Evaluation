def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    results = []
    for j in range(m):
        eater_person = -1
        for i in range(n):
            if b[j] >= a[i]:
                eater_person = i + 1
                break
        results.append(eater_person)
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()