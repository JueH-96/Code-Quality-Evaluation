def solve():
    n = int(input())
    dishes = [input() for _ in range(n)]
    
    for i in range(n - 1):
        if dishes[i] == "sweet" and dishes[i+1] == "sweet":
            print("No")
            return
    print("Yes")

solve()