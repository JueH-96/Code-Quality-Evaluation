# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    M = int(data[idx])
    idx += 1
    B = list(map(int, data[idx:idx+M]))
    idx += M
    L = int(data[idx])
    idx += 1
    C = list(map(int, data[idx:idx+L]))
    idx += L
    Q = int(data[idx])
    idx += 1
    X = list(map(int, data[idx:idx+Q]))
    
    # Precompute all possible sums of A and B
    sum_AB = set()
    for a in A:
        for b in B:
            sum_AB.add(a + b)
    
    # Precompute all possible sums of C
    sum_C = set(C)
    
    # For each X_i, check if there exists a sum_ab in sum_AB and a c in sum_C such that sum_ab + c == X_i
    for x in X:
        found = False
        for sum_ab in sum_AB:
            if (x - sum_ab) in sum_C:
                found = True
                break
        if found:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()