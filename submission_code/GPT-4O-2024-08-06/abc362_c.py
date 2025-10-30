# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    L = []
    R = []
    
    index = 1
    for _ in range(N):
        L.append(int(data[index]))
        R.append(int(data[index + 1]))
        index += 2
    
    min_sum = sum(L)
    max_sum = sum(R)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    # We need to make the sum exactly 0
    # Start with the minimum sum configuration
    current_sum = min_sum
    X = L[:]
    
    # We need to increase the current_sum to 0
    for i in range(N):
        if current_sum == 0:
            break
        # How much can we increase X[i] without exceeding R[i]
        increase = min(R[i] - X[i], -current_sum)
        X[i] += increase
        current_sum += increase
    
    if current_sum == 0:
        print("Yes")
        print(" ".join(map(str, X)))
    else:
        print("No")

main()