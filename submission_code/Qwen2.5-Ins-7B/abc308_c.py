# N = int(input())
# data = [list(map(int, input().split())) for _ in range(N)]
# people = sorted([(a/b, -i) for i, (a, b) in enumerate(data)])
# for _, i in people:
#     print(-i, end=' ')
N = int(input())
data = [list(map(int, input().split())) for _ in range(N)]
people = sorted([(a/b, -i) for i, (a, b) in enumerate(data)], reverse=True)
for _, i in people:
    print(-i, end=' ')