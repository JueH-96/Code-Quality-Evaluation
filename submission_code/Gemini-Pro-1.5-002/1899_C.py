# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_sum = -float('inf')

    for i in range(n):
        current_sum = 0
        current_subarray = []
        for j in range(i, n):
            valid = True
            if current_subarray:
                if current_subarray[-1] % 2 == a[j] % 2:
                    valid = False
            
            if valid:
                current_subarray.append(a[j])
                current_sum += a[j]
                max_sum = max(max_sum, current_sum)
            else:
                break

    print(max_sum)


t = int(input())
for _ in range(t):
    solve()