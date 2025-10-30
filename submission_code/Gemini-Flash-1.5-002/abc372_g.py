# YOUR CODE HERE
def solve():
    N = int(input())
    constraints = []
    for _ in range(N):
        A, B, C = map(int, input().split())
        constraints.append((A, B, C))

    count = 0
    for x in range(1, 1001):  # Adjust the range as needed
        for y in range(1, 1001):  # Adjust the range as needed
            valid = True
            for A, B, C in constraints:
                if A * x + B * y >= C:
                    valid = False
                    break
            if valid:
                count += 1
    print(count)


T = int(input())
for _ in range(T):
    solve()