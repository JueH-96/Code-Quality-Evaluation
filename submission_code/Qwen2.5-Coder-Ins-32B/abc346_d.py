# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    C = list(map(int, input[2:]))
    
    min_cost = float('inf')
    
    # Try to make the i-th and (i+1)-th characters the same for each i
    for i in range(N - 1):
        cost = 0
        # Flip characters to the left of i if needed
        for j in range(i):
            if S[j] == S[i]:
                cost += C[j]
        # Flip characters to the right of i+1 if needed
        for j in range(i + 2, N):
            if S[j] == S[i + 1]:
                cost += C[j]
        # Check if flipping the i-th or (i+1)-th character is needed
        if S[i] != S[i + 1]:
            cost += min(C[i], C[i + 1])
        min_cost = min(min_cost, cost)
    
    print(min_cost)

if __name__ == "__main__":
    main()