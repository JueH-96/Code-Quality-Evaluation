# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    import sys
    input = sys.stdin.read().split()
    idx =0
    T = int(input[idx])
    idx +=1
    results = []
    for _ in range(T):
        N = int(input[idx])
        idx +=1
        A = list(map(int, input[idx:idx+N]))
        idx +=N
        prev_B = A[-1]
        residual =0
        feasible = True
        for i in range(N-2, -1, -1):
            B_i = min(A[i] + residual, prev_B)
            residual = A[i] + residual - B_i
            prev_B = B_i
        if residual ==0:
            results.append("Yes")
        else:
            results.append("No")
    print('
'.join(results))

if __name__ == "__main__":
    main()