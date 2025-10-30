# YOUR CODE HERE
import sys

def main():
    import sys
    import math

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    courses = []
    idx = 1
    for _ in range(N):
        X = int(data[idx])
        Y = int(data[idx+1])
        courses.append( (X, Y) )
        idx +=2
    
    dp_H = 0
    dp_U = -math.inf
    for X, Y in courses:
        # Compute new dp_H
        new_dp_H_candidates = [dp_H]
        if X ==0:
            new_dp_H_candidates.append(dp_H + Y)
            if dp_U > -math.inf:
                new_dp_H_candidates.append(dp_U + Y)
        new_dp_H = max(new_dp_H_candidates)
        
        # Compute new dp_U
        new_dp_U_candidates = [dp_U]
        if X ==1:
            new_dp_U_candidates.append(dp_H + Y)
        new_dp_U = max(new_dp_U_candidates)
        
        dp_H, dp_U = new_dp_H, new_dp_U
    
    result = max(dp_H, dp_U, 0)
    print(result)

if __name__ == "__main__":
    main()