import sys
import math

def main():
    N = int(sys.stdin.readline())
    buildings = []
    for _ in range(N):
        X, H = map(int, sys.stdin.readline().split())
        buildings.append((X, H))
    
    # Calculate the maximum height h where not all buildings are visible
    # We need to find the minimum h where all buildings are visible
    # The answer is the maximum h where at least one building is not visible
    
    # For each building, calculate the minimum height h where it is visible
    # The height h must be such that the line from (0, h) to (X_i, H_i) does not intersect any other building
    
    # For each building i, the line from (0, h) to (X_i, H_i) must be above all other buildings
    # The slope of the line is (H_i - h) / X_i
    # For any building j, the line must be above (X_j, H_j)
    # So, (H_i - h) / X_i * X_j + h >= H_j
    # Rearranged: h >= (H_j * X_i - H_i * X_j) / (X_i - X_j)
    
    # To find the minimum h where all buildings are visible, we need to find the maximum of the lower bounds for h for each building
    
    # For each building i, compute the maximum lower bound for h from all other buildings j
    # Then, the overall minimum h is the maximum of these lower bounds
    
    # Initialize the overall minimum h as 0
    min_h = 0
    
    for i in range(N):
        X_i, H_i = buildings[i]
        max_lower_bound = 0
        for j in range(N):
            if j == i:
                continue
            X_j, H_j = buildings[j]
            if X_i == X_j:
                continue
            # Calculate the lower bound for h
            numerator = H_j * X_i - H_i * X_j
            denominator = X_i - X_j
            if denominator == 0:
                continue
            lower_bound = numerator / denominator
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
        # The minimum h for building i is max_lower_bound
        # The overall minimum h is the maximum of all these min_h_i
        if max_lower_bound > min_h:
            min_h = max_lower_bound
    
    # Now, the answer is the maximum height where not all buildings are visible
    # Which is the minimum h where all buildings are visible minus a small epsilon
    # But since we need the maximum height where not all are visible, it's the minimum h minus epsilon
    
    # However, the problem asks for the maximum height where not all buildings are visible
    # So, if the minimum h is 0, then the answer is -1
    if min_h == 0:
        print(-1)
    else:
        # Otherwise, the answer is the minimum h minus a small epsilon
        # But since the problem allows for an absolute or relative error of at most 1e-9, we can directly output min_h
        # Because the maximum height where not all buildings are visible is just below min_h
        # So, the answer is min_h
        print("{0:.20f}".format(min_h))

if __name__ == "__main__":
    main()