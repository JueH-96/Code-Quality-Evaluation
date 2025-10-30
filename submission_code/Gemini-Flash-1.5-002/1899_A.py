# YOUR CODE HERE
def solve():
    n = int(input())
    
    if n % 3 == 0:
        print("First")
        return

    for i in range(10):
        if (n + 1) % 3 == 0:
            print("First")
            return
        if (n -1) % 3 == 0:
            print("First")
            return
        n +=1
        if (n + 1) % 3 == 0:
            print("First")
            return
        if (n -1) % 3 == 0:
            print("First")
            return
        n -=2

    print("Second")


t = int(input())
for _ in range(t):
    solve()