def solve():
    n = int(input())
    dishes = [input() for _ in range(n)]
    
    sweet_count = 0
    for i in range(n):
        if dishes[i] == 'sweet':
            sweet_count += 1
        else:
            sweet_count = 0
        
        if sweet_count > 1 and i < n -1:
            print("No")
            return
    
    print("Yes")

solve()