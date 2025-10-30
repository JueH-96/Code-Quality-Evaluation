import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    T = list(data[3])
    
    # We will iterate over each character in T and replace the smallest possible character in S
    # that is less than the current character in T to maximize the number.
    for k in range(M):
        t_char = T[k]
        # Find the best position in S to replace with t_char
        for i in range(N):
            if S[i] < t_char:
                S[i] = t_char
                break
    
    # Join the list S to form the final string and print it
    print(''.join(S))