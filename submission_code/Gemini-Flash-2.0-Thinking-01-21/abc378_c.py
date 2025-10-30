def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    for i in range(n):
        found_index = -1
        for j in range(i - 1, -1, -1):
            if a[j] == a[i]:
                found_index = j + 1
                break
        b.append(found_index)
    print(*(b))

if __name__ == "__main__":
    solve()