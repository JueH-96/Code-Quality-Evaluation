# YOUR CODE HERE
def solve():
    s = input()
    if s == "abc":
        print("YES")
    elif s == "acb":
        print("YES")
    elif s == "bac":
        print("YES")
    elif s == "bca":
        print("NO")
    elif s == "cab":
        print("NO")
    elif s == "cba":
        print("YES")

t = int(input())
for _ in range(t):
    solve()