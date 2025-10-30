def solve():
    n = int(input())
    s_types = [input() for _ in range(n)]
    
    if n <= 1:
        print("Yes")
        return
        
    for i in range(n - 1):
        if s_types[i] == "sweet" and s_types[i+1] == "sweet":
            print("No")
            return
            
    print("Yes")

if __name__ == '__main__':
    solve()