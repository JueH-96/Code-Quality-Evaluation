# YOUR CODE HERE
def solve():
    s, t = input().split()
    n = len(s)
    m = len(t)

    for w in range(1, n):
        for c in range(1, w + 1):
            res = ""
            for i in range(0, n, w):
                if i + c < n and i + w <= n:
                    
                    res += s[i + c]
                elif i + c < n :
                    if c <= len(s[i:]):
                        res += s[i+c]

            if res == t:
                print("Yes")
                return

    print("No")

solve()