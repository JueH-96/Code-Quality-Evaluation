def min_operations(n, k, s):
    count = 0
    covered_up_to = -1
    for i in range(n):
        if i > covered_up_to and s[i] == 'B':
            count += 1
            covered_up_to = i + k - 1
    return count

# Read input and process each test case
t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    print(min_operations(n, k, s))