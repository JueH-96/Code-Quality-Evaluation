# YOUR CODE HERE
def solve():
    s = input()
    t = input()
    n = len(s)
    
    if s == t:
        print(0)
        return

    diff_indices = []
    for i in range(n):
        if s[i] != t[i]:
            diff_indices.append(i)
    
    ans = []
    curr_s = list(s)
    
    for i in diff_indices:
        curr_s[i] = t[i]
        ans.append("".join(curr_s))
        
    print(len(ans))
    for x in ans:
        print(x)

solve()