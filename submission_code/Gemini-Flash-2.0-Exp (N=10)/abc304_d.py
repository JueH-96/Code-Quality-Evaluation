def solve():
    W, H = map(int, input().split())
    N = int(input())
    strawberries = []
    for _ in range(N):
        strawberries.append(list(map(int, input().split())))
    A = int(input())
    a = list(map(int, input().split()))
    B = int(input())
    b = list(map(int, input().split()))

    a = [0] + a + [W]
    b = [0] + b + [H]

    counts = []
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            count = 0
            for p, q in strawberries:
                if a[i] < p < a[i+1] and b[j] < q < b[j+1]:
                    count += 1
            counts.append(count)
    
    print(min(counts), max(counts))

solve()