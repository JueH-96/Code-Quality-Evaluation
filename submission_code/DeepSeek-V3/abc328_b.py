# YOUR CODE HERE
def is_repdigit(x):
    s = str(x)
    return all(c == s[0] for c in s)

def count_repdigit_days(N, D):
    count = 0
    for i in range(1, N+1):
        if is_repdigit(i):
            for j in range(1, D[i-1]+1):
                if is_repdigit(j):
                    count += 1
    return count

# Read input
N = int(input())
D = list(map(int, input().split()))

# Compute and print the result
print(count_repdigit_days(N, D))