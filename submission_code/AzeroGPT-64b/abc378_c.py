from bisect import bisect_left
from typing import Any, Dict

N = int(input())
A = tuple(map(int, input().split()))
# cnter of value by most recent index
memo: Dict[int, int] = {}
# most recent indices sorted
sorted_idx = [0]
ans = []
for i, a in enumerate(A):
    if a in memo:
        ans.append(memo[a])
    else:
        ans.append(-1)

    del_idx = bisect_left(sorted_idx, memo[a]) if a in memo else None
    memo[a] = i
    if del_idx is not None:
        del sorted_idx[del_idx]
    bisect_insert_idx = bisect_left(sorted_idx, i)
    sorted_idx.insert(bisect_insert_idx, i)

print(*ans)