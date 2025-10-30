# YOUR CODE HERE
def nand(a, b):
    return 1 if a + b != 2 else 0

def solve(n, s):
    s = [int(c) for c in s]
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + s[i - 1]
    
    result = 0
    for i in range(n):
        count_1 = prefix[n] - prefix[i]
        count_0 = (n - i) - count_1
        if s[i] == 1:
            result += count_0
        else:
            result += count_1
    return result

n = int(input())
s = input().strip()
print(solve(n, s))