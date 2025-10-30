import sys

def solve():
    N = int(input())
    A = input().split()
    
    # Apply the operation N times to get A'
    for _ in range(N):
        B = []
        for i in range(0, len(A), 3):
            count0 = A[i:i+3].count('0')
            count1 = A[i:i+3].count('1')
            B.append('0' if count0 > count1 else '1')
        A = B
    
    # Determine the minimum number of changes to A to change A'
    target = '0' if A[0] == '1' else '1'
    changes = 0
    for a in A:
        if a != target:
            changes += 1
    
    print(changes)

solve()