# YOUR CODE HERE
def can_eat_all_dishes():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    dishes = data[1:]
    
    for i in range(1, N):
        if dishes[i] == "sweet" and dishes[i-1] == "sweet":
            print("No")
            return
    
    print("Yes")

can_eat_all_dishes()