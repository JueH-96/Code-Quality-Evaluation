def solve():
    n_str = input()
    if len(n_str) <= 1:
        print("Yes")
        return
    
    is_321_like = True
    for i in range(len(n_str) - 1):
        digit1 = int(n_str[i])
        digit2 = int(n_str[i+1])
        if digit1 <= digit2:
            is_321_like = False
            break
            
    if is_321_like:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()