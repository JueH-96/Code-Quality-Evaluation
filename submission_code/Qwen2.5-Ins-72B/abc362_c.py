# YOUR CODE HERE
import sys

def find_sequence(N, ranges):
    # Initialize the sequence with the minimum values
    X = [L for L, R in ranges]
    total_sum = sum(X)
    
    # Adjust the sequence to make the sum zero
    for i in range(N):
        L, R = ranges[i]
        if total_sum > 0:
            # If the total sum is positive, try to decrease it
            if X[i] > L:
                decrease = min(total_sum, X[i] - L)
                X[i] -= decrease
                total_sum -= decrease
        else:
            # If the total sum is negative, try to increase it
            if X[i] < R:
                increase = min(-total_sum, R - X[i])
                X[i] += increase
                total_sum += increase
    
    # Check if the total sum is zero
    if total_sum == 0:
        return X
    else:
        return None

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    ranges = []
    index = 1
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        ranges.append((L, R))
        index += 2
    
    X = find_sequence(N, ranges)
    
    if X is None:
        print("No")
    else:
        print("Yes")
        print(" ".join(map(str, X)))

if __name__ == "__main__":
    main()