N = int(input())
AB = [list(map(int, input().split())) for _ in range(N)]
AB = [[(a / (a + b)), i + 1] for i, (a, b) in enumerate(AB)]
AB.sort(key=lambda x: (-x[0], x[1]))
print(*[i for _, i in AB])