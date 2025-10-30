# YOUR CODE HERE
def solve():
    N = int(input())
    H = list(map(int, input().split()))
    
    T = 0
    i = 0
    
    while i < N:
        T += 1
        if T % 3 == 0:
            H[i] -= 3
        else:
            H[i] -= 1
        
        if H[i] <= 0:
            i += 1
    
    print(T)

solve()