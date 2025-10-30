def solve():
    n = int(input())
    s = input()
    if n % 2 == 0:
        print("No")
        return
    k = (n - 1) // 2
    if k < 0:
        k = 0
    
    for i in range(k):
        if s[i] != '1':
            print("No")
            return
            
    if s[k] != '/':
        print("No")
        return
        
    for i in range(k + 1, n):
        if s[i] != '2':
            print("No")
            return
            
    print("Yes")

if __name__ == '__main__':
    solve()