# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = list(map(int, data[N+1:2*N]))
    T = list(map(int, data[2*N+1:3*N-1]))
    
    money = A[0]
    for i in range(N-1):
        exchange_rate = S[i] / T[i]
        while money >= S[i]:
            exchange_amount = money // S[i]
            A[i+1] += exchange_amount * T[i]
            money -= exchange_amount * S[i]
        money = A[i+1]
    
    print(money)

solve()