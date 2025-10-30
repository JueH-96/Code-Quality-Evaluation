# YOUR CODE HERE
def solve():
    s = input()
    if s == "abc":
        print("YES")
        return
    
    if s == "acb":
        print("YES")
        return
    
    if s == "bac":
        print("YES")
        return
    
    if s == "bca":
        print("NO")
        return
    
    if s == "cab":
        print("NO")
        return
    
    if s == "cba":
        print("YES")
        return

t = int(input())
for _ in range(t):
    solve()