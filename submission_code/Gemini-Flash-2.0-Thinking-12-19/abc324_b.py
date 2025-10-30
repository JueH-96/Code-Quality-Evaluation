def solve():
    n = int(input())
    if n <= 0:
        print("No")
        return
    
    while n % 2 == 0:
        n //= 2
        
    while n % 3 == 0:
        n //= 3
        
    if n == 1:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()