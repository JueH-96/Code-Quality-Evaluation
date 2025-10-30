# YOUR CODE HERE
S = input()
T = input()

def solve():
    n = len(S)
    
    # Check for subsequence of length 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                subsequence = S[i] + S[j] + S[k]
                if subsequence.upper() == T:
                    print("Yes")
                    return

    # Check for subsequence of length 2 + 'X'
    for i in range(n):
        for j in range(i + 1, n):
            subsequence = S[i] + S[j]
            if subsequence.upper() + "X" == T:
                print("Yes")
                return

    print("No")

solve()