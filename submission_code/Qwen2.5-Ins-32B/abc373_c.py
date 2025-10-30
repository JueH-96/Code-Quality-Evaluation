import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    max_A = max(A)
    max_B = max(B)
    
    print(max_A + max_B)

if __name__ == "__main__":
    solve()