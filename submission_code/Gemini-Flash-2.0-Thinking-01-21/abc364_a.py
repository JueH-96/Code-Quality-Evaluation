def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    if n <= 1:
        print("Yes")
        return
    
    first_sick_dish_index = -1
    for i in range(1, n):
        if s[i-1] == 'sweet' and s[i] == 'sweet':
            first_sick_dish_index = i + 1
            break
            
    if first_sick_dish_index == -1:
        print("Yes")
    elif first_sick_dish_index == n + 1: # This should not happen, index is at most n
        print("Yes") 
    elif first_sick_dish_index == n:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()