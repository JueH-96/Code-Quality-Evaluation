import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    is_strictly_increasing = True
    for i in range(N - 1):
        if A[i] >= A[i+1]:
            is_strictly_increasing = False
            break
    
    if is_strictly_increasing:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()