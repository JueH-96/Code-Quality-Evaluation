import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    L = []
    R = []
    
    for i in range(N):
        L.append(int(data[2*i + 1]))
        R.append(int(data[2*i + 2]))
    
    # We need to find a sequence X such that L_i <= X_i <= R_i and sum(X) = 0
    # Start by setting X_i = L_i and adjust if necessary
    X = L[:]
    total_sum = sum(X)
    
    if total_sum == 0:
        print("Yes")
        print(" ".join(map(str, X)))
        return
    
    # If the total sum is not zero, we need to adjust the values of X
    for i in range(N):
        if total_sum == 0:
            break
        if total_sum > 0:
            # We need to decrease some X_i
            if X[i] < R[i]:
                # We can increase X[i] up to R[i]
                max_decrease = min(total_sum, R[i] - X[i])
                X[i] += max_decrease
                total_sum -= max_decrease
        else:
            # We need to increase some X_i
            if X[i] > L[i]:
                # We can decrease X[i] down to L[i]
                max_increase = min(-total_sum, X[i] - L[i])
                X[i] -= max_increase
                total_sum += max_increase
    
    if total_sum == 0:
        print("Yes")
        print(" ".join(map(str, X)))
    else:
        print("No")