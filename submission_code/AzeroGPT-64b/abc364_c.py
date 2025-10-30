import sys
import bisect

N, X, Y = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

name = list(zip(A, B))
sort_list = []
sort_list_acc = [0, 0]

# Consider case only taking the first i elements (greedy)
for i in range(N):
    sort_list.append((sort_list_acc[0] + name[i][0], sort_list_acc[1] + name[i][1]))
    sort_list_acc = (sort_list[-1][0], sort_list[-1][1])

# Now since the cases are greedy, simply take last case less than X, Y
max_idx = bisect.bisect(sort_list, (X, Y)) - 1
if max_idx == -1:
    print(0)
else:
    print(max_idx + 1)