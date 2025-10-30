def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    k = int(input())
    queries = []
    for _ in range(k):
        queries.append(list(map(int, input().split())))

    for x_k, y_k in queries:
        current_sum = 0
        for i in range(x_k):
            for j in range(y_k):
                current_sum += abs(a[i] - b[j])
        print(current_sum)

solve()