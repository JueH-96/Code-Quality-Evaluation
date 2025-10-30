# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:3+N]))
    
    # Separate ants into left-moving and right-moving
    left = []
    right = []
    for i in range(N):
        if S[i] == '0':
            left.append(X[i])
        else:
            right.append(X[i])
    
    # Sort the positions
    left.sort()
    right.sort()
    
    # Initialize the count of pairs
    count = 0
    
    # For each left-moving ant, find the number of right-moving ants that are to its left
    # and will pass it within time T
    for x in left:
        # The right-moving ants must be to the left of x and moving towards x
        # Their position must be >= x - T
        low = 0
        high = len(right) - 1
        res = -1
        while low <= high:
            mid = (low + high) // 2
            if right[mid] <= x:
                res = mid
                low = mid + 1
            else:
                high = mid - 1
        if res != -1:
            # Now, among these, find those that are >= x - T
            low2 = 0
            high2 = res
            res2 = -1
            while low2 <= high2:
                mid2 = (low2 + high2) // 2
                if right[mid2] >= x - T:
                    res2 = mid2
                    high2 = mid2 - 1
                else:
                    low2 = mid2 + 1
            if res2 != -1:
                count += (res - res2 + 1)
    
    # For each right-moving ant, find the number of left-moving ants that are to its right
    # and will pass it within time T
    for x in right:
        # The left-moving ants must be to the right of x and moving towards x
        # Their position must be <= x + T
        low = 0
        high = len(left) - 1
        res = -1
        while low <= high:
            mid = (low + high) // 2
            if left[mid] >= x:
                res = mid
                high = mid - 1
            else:
                low = mid + 1
        if res != -1:
            # Now, among these, find those that are <= x + T
            low2 = res
            high2 = len(left) - 1
            res2 = -1
            while low2 <= high2:
                mid2 = (low2 + high2) // 2
                if left[mid2] <= x + T:
                    res2 = mid2
                    low2 = mid2 + 1
                else:
                    high2 = mid2 - 1
            if res2 != -1:
                count += (res2 - res + 1)
    
    print(count)

if __name__ == "__main__":
    main()