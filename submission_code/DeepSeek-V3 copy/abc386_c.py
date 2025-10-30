# YOUR CODE HERE
K = int(input())
S = input().strip()
T = input().strip()

if S == T:
    print("Yes")
    exit()

len_S = len(S)
len_T = len(T)

if abs(len_S - len_T) > 1:
    print("No")
    exit()

if len_S == len_T:
    # Only replacement is possible
    diff = 0
    for i in range(len_S):
        if S[i] != T[i]:
            diff += 1
            if diff > 1:
                print("No")
                exit()
    print("Yes")
elif len_S < len_T:
    # Insertion is possible
    # Check if T can be obtained by inserting one character into S
    i = j = 0
    diff = 0
    while i < len_S and j < len_T:
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
    # Deletion is possible
    # Check if T can be obtained by deleting one character from S
    i = j = 0
    diff = 0
    while i < len_S and j < len_T:
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