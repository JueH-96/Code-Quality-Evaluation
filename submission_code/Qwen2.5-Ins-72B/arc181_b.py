# YOUR CODE HERE
import sys

def can_form_same_string(S, X, Y):
    # Count the number of 0s and 1s in X and Y
    count_X_0 = X.count('0')
    count_X_1 = X.count('1')
    count_Y_0 = Y.count('0')
    count_Y_1 = Y.count('1')
    
    # If the number of 0s and 1s are the same, T can be empty
    if count_X_0 == count_Y_0 and count_X_1 == count_Y_1:
        return True
    
    # If the number of 0s and 1s are different, check if S can be repeated to match the difference
    if count_X_0 == count_Y_0:
        if (count_X_1 - count_Y_1) * len(S) == 0:
            return True
    elif count_X_1 == count_Y_1:
        if (count_X_0 - count_Y_0) * len(S) == 0:
            return True
    else:
        return False
    
    return False

def main():
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    index = 1
    results = []
    
    for _ in range(t):
        S = data[index]
        X = data[index + 1]
        Y = data[index + 2]
        index += 3
        
        if can_form_same_string(S, X, Y):
            results.append("Yes")
        else:
            results.append("No")
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()