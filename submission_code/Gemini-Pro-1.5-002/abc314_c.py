# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    s = list(input())
    c = list(map(int, input().split()))

    for i in range(1, m + 1):
        indices = [j for j in range(n) if c[j] == i]
        temp = [s[j] for j in indices]
        shifted_temp = [temp[-1]] + temp[:-1]
        for k in range(len(indices)):
            s[indices[k]] = shifted_temp[k]

    print("".join(s))

solve()