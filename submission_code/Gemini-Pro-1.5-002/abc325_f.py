# YOUR CODE HERE
def solve():
    n = int(input())
    d = list(map(int, input().split()))
    l1, c1, k1 = map(int, input().split())
    l2, c2, k2 = map(int, input().split())

    total_length = sum(d)

    min_cost = float('inf')

    for num1 in range(min(k1 + 1, total_length // l1 + 2)):
        remaining_length = total_length - num1 * l1
        if remaining_length <= 0:
            min_cost = min(min_cost, num1 * c1)
            continue
        
        num2 = (remaining_length + l2 - 1) // l2
        if num2 > k2:
            continue
        
        min_cost = min(min_cost, num1 * c1 + num2 * c2)

    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()