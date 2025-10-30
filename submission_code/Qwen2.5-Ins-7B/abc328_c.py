# N, Q = map(int, input().split())
# S = input()
# queries = [tuple(map(int, input().split())) for _ in range(Q)]

# ans = []
# for l, r in queries:
#     count = 0
#     for i in range(l-1, r-1):
#         if S[i] == S[i+1]:
#             count += 1
#     ans.append(count)

# for a in ans:
#     print(a)

N, Q = map(int, input().split())
S = input()
queries = [tuple(map(int, input().split())) for _ in range(Q)]

for l, r in queries:
    count = 0
    for i in range(l-1, r-1):
        if S[i] == S[i+1]:
            count += 1
    print(count)