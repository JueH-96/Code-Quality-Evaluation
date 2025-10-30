# YOUR CODE HERE

def min_total_payment(N, P, Q, D):
    D.sort()
    if Q < P:
        return Q + D[0]
    else:
        return P + D[0]

N = int(input().split()[0])
P = int(input().split()[0])
Q = int(input().split()[0])
D = list(map(int, input().split()))

print(min_total_payment(N, P, Q, D))