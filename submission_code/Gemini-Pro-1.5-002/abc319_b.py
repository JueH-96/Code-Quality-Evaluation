# YOUR CODE HERE
def solve():
    n = int(input())
    ans = ""
    for i in range(n + 1):
        found = False
        smallest_j = -1
        for j in range(1, 10):
            if n % j == 0:
                if i % (n // j) == 0:
                    if not found:
                        smallest_j = j
                        found = True
                    else:
                        smallest_j = min(smallest_j, j)
        if found:
            ans += str(smallest_j)
        else:
            ans += "-"
    print(ans)

solve()