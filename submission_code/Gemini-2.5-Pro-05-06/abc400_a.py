# YOUR CODE HERE
def solve():
    A = int(input())
    
    total_people = 400
    
    if total_people % A == 0:
        B = total_people // A
        # B must be a positive integer.
        # Since total_people (400) is positive and A is positive (1 <= A <= 400),
        # if B is an integer, it will also be positive.
        print(B)
    else:
        print(-1)

if __name__ == "__main__":
    solve()