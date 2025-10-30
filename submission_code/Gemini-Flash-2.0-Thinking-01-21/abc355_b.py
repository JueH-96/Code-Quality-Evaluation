def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    set_a = set(a)
    combined_list = a + b
    c = sorted(combined_list)
    
    found_consecutive_a_pair = False
    for i in range(len(c) - 1):
        if c[i] in set_a and c[i+1] in set_a:
            found_consecutive_a_pair = True
            break
            
    if found_consecutive_a_pair:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()