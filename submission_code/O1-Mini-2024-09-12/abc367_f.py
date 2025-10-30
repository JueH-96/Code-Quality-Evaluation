import sys, random

def solve():
    import sys, sys
    from sys import stdin
    def input():
        return sys.stdin.read()
    data = input().split()
    pos = 0
    N, Q = int(data[pos]), int(data[pos+1]); pos +=2
    A = list(map(int, data[pos:pos+N])); pos +=N
    B = list(map(int, data[pos:pos+N])); pos +=N

    # Assign two random 64-bit numbers to each value
    random.seed(123456789)
    rand1 = [0]*(N+1)
    rand2 = [0]*(N+1)
    for i in range(1, N+1):
        rand1[i] = random.getrandbits(64)
        rand2[i] = random.getrandbits(64)

    # Compute prefix sums for A
    prefA1 = [0]*(N+1)
    prefA2 = [0]*(N+1)
    for i in range(1, N+1):
        prefA1[i] = prefA1[i-1] + rand1[A[i-1]]
        prefA2[i] = prefA2[i-1] + rand2[A[i-1]]

    # Compute prefix sums for B
    prefB1 = [0]*(N+1)
    prefB2 = [0]*(N+1)
    for i in range(1, N+1):
        prefB1[i] = prefB1[i-1] + rand1[B[i-1]]
        prefB2[i] = prefB2[i-1] + rand2[B[i-1]]

    output = []
    for _ in range(Q):
        if pos+4 > len(data):
            l, r, L, R = 1,1,1,1
        else:
            l = int(data[pos]); pos +=1
            r = int(data[pos]); pos +=1
            L = int(data[pos]); pos +=1
            R = int(data[pos]); pos +=1
        lenA = r - l +1
        lenB = R - L +1
        if lenA != lenB:
            output.append("No")
            continue
        sumA1 = prefA1[r] - prefA1[l-1]
        sumA2 = prefA2[r] - prefA2[l-1]
        sumB1 = prefB1[R] - prefB1[L-1]
        sumB2 = prefB2[R] - prefB2[L-1]
        if sumA1 == sumB1 and sumA2 == sumB2:
            output.append("Yes")
        else:
            output.append("No")
    print('
'.join(output))