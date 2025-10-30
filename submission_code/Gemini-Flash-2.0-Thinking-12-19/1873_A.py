def solve():
    s = input()
    if s == "abc":
        print("YES")
        return
    
    s_list = list(s)
    
    # Swap 1st and 2nd
    s_list_12 = list(s_list)
    s_list_12[0], s_list_12[1] = s_list_12[1], s_list_12[0]
    if "".join(s_list_12) == "abc":
        print("YES")
        return
        
    # Swap 1st and 3rd
    s_list_13 = list(s_list)
    s_list_13[0], s_list_13[2] = s_list_13[2], s_list_13[0]
    if "".join(s_list_13) == "abc":
        print("YES")
        return
        
    # Swap 2nd and 3rd
    s_list_23 = list(s_list)
    s_list_23[1], s_list_23[2] = s_list_23[2], s_list_23[1]
    if "".join(s_list_23) == "abc":
        print("YES")
        return
        
    print("NO")

t = int(input())
for _ in range(t):
    solve()