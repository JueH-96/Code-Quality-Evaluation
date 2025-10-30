import sys

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Sort the lists to match the largest B with the smallest possible A
    A.sort()
    B.sort()
    
    # Check if it's possible to satisfy the condition
    if A[M-1] < B[M-1]:
        print(-1)
        return
    
    # Calculate the minimum total amount of money needed
    total_cost = sum(A[:M])
    print(total_cost)

if __name__ == "__main__":
    solve()