# YOUR CODE HERE
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
    
    # Calculate the sum of L_i and R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
    
    # Initialize X with L_i
    X = L.copy()
    total = sum_L
    
    # Adjust X to make the sum zero
    for i in range(N):
        if total < 0:
            add = min(R[i] - L[i], -total)
            X[i] += add
            total += add
    
    if total == 0:
        print("Yes")
        print(' '.join(map(str, X)))
    else:
        print("No")

if __name__ == "__main__":
    main()