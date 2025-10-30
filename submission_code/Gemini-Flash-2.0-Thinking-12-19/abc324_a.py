def solve():
    n = int(input())
    a_list_str = input().split()
    a_list = [int(x) for x in a_list_str]
    
    if n <= 0:
        print("Yes") # Technically, for n=0 or n=1, all are considered equal. But problem constraint is n>=2.
        return
        
    first_element = a_list[0]
    all_equal = True
    for i in range(1, n):
        if a_list[i] != first_element:
            all_equal = False
            break
            
    if all_equal:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()