# YOUR CODE HERE
import sys

def can_transform(S, T):
    N = len(S)
    M = len(T)
    
    # Create a list to track which parts of S are covered by T
    covered = [False] * N
    
    # Try to cover S with T
    for i in range(N - M + 1):
        if all(S[i + j] == T[j] or covered[i + j] for j in range(M)):
            for j in range(M):
                covered[i + j] = True
    
    # Check if the entire S is covered
    return all(covered)

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    S = input[2]
    T = input[3]
    
    if can_transform(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()