def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Check if all positions are zeros and N is odd
    if all(a == 0 for a in A) and N % 2 == 1:
        print("No")
    else:
        print("Yes")

solve()