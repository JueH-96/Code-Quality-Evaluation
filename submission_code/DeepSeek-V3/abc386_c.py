# YOUR CODE HERE
K = int(input())
S = input().strip()
T = input().strip()

if S == T:
    print("Yes")
    exit()

if K == 0:
    print("No")
    exit()

n = len(S)
m = len(T)

if abs(n - m) > 1:
    print("No")
    exit()

if n == m:
    # Check if they can be made equal with one replacement
    diff = 0
    for i in range(n):
        if S[i] != T[i]:
            diff += 1
            if diff > 1:
                print("No")
                exit()
    print("Yes")
elif n < m:
    # Check if S can be transformed into T with one insertion
    i = j = 0
    diff = 0
    while i < n and j < m:
        if S[i] == T[j]:
            i += 1
            j += 1
        else:
            diff += 1
            j += 1
            if diff > 1:
                print("No")
                exit()
    print("Yes")
else:
    # Check if S can be transformed into T with one deletion
    i = j = 0
    diff = 0
    while i < n and j < m:
        if S[i] == T[j]:
            i += 1
            j += 1
        else:
            diff += 1
            i += 1
            if diff > 1:
                print("No")
                exit()
    print("Yes")