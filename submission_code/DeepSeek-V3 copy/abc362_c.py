def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    L = []
    R = []
    for i in range(N):
        L.append(int(data[1 + 2*i]))
        R.append(int(data[2 + 2*i]))
    
    # Calculate the minimum and maximum possible sum
    min_sum = sum(L)
    max_sum = sum(R)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    # We need to find X_i such that sum(X_i) = 0 and L_i <= X_i <= R_i
    # We can start by setting X_i = L_i, then adjust to reach sum 0
    
    X = L.copy()
    current_sum = sum(X)
    
    # We need to increase the sum by -current_sum
    # So we need to find which X_i can be increased
    # We will iterate through the X_i and increase them as much as possible
    
    for i in range(N):
        if current_sum == 0:
            break
        # Calculate how much we can increase X_i
        delta = min(R[i] - X[i], -current_sum)
        X[i] += delta
        current_sum += delta
    
    if current_sum == 0:
        print("Yes")
        print(' '.join(map(str, X)))
    else:
        print("No")

if __name__ == "__main__":
    main()