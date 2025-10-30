import heapq

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    min_product = float('inf')

    for i in range(n):
        max_a = a[i]
        for m in range(n):
            if a[m] == max_a:
                eligible_b_values = []
                for j in range(n):
                    if a[j] <= max_a and j != m:
                        eligible_b_values.append(b[j])

                if len(eligible_b_values) >= k - 1:
                    smallest_b = heapq.nsmallest(k - 1, eligible_b_values)
                    current_sum_b = sum(smallest_b) + b[m]
                    current_product = max_a * current_sum_b
                    min_product = min(min_product, current_product)

    print(min_product)

t = int(input())
for _ in range(t):
    solve()