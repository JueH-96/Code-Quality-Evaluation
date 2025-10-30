# YOUR CODE HERE
from itertools import combinations

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    min_diff = float('inf')

    for combo in combinations(range(n), k):
        temp_a = []
        for i in range(n):
            if i not in combo:
                temp_a.append(a[i])
        
        if not temp_a:
            continue

        min_diff = min(min_diff, max(temp_a) - min(temp_a))

    print(min_diff)

solve()