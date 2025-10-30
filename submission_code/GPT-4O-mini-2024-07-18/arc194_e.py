def can_transform(N, X, Y, S, T):
    # Create a list to keep track of the difference between S and T
    diff = [0] * N
    
    # Calculate the difference between S and T
    for i in range(N):
        diff[i] = 1 if S[i] != T[i] else 0
    
    # We will use two pointers to track the segments of 1s and 0s
    i = 0
    while i < N:
        if diff[i] == 1:
            # We found a segment that needs to be changed
            # Check if we can perform an operation
            # We need to find the length of the segment of 1s
            start = i
            while i < N and diff[i] == 1:
                i += 1
            length_of_ones = i - start
            
            # Now we need to check if we can cover this segment with operations
            if length_of_ones < Y or (length_of_ones - Y) % (X + Y) != 0:
                return "No"
        else:
            i += 1
    
    # If we reach here, it means we can transform S into T
    return "Yes"

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, X, Y = map(int, data[0].split())
    S = data[1].strip()
    T = data[2].strip()
    
    result = can_transform(N, X, Y, S, T)
    print(result)

if __name__ == "__main__":
    main()