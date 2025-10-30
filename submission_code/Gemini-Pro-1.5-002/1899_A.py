# YOUR CODE HERE
def solve():
    n = int(input())
    
    if n % 3 == 0:
        print("Second")
        return

    if n % 3 == 1:
        print("First")
        return

    if n % 3 == 2:
        print("First")
        return


t = int(input())
for _ in range(t):
    solve()