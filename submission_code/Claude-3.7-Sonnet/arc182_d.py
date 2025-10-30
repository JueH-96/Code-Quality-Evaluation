def min_operations_to_transform(N, M, A, B):
    total_ops = 0
    
    for i in range(N):
        # Determine the values to avoid (adjacent elements)
        avoid = set()
        if i > 0:
            avoid.add(A[i-1])
        if i < N-1:
            avoid.add(A[i+1])
        
        # If B[i] is one of the values to avoid, it's not possible
        if B[i] in avoid:
            return -1
        
        # If A[i] is already equal to B[i], no operations needed
        if A[i] == B[i]:
            continue
        
        # Compute the clockwise and counterclockwise distances
        cw_dist = (B[i] - A[i] + M) % M
        ccw_dist = (A[i] - B[i] + M) % M
        
        # Check if the clockwise path has any conflicts
        cw_conflict = False
        for val in avoid:
            if val != A[i] and val != B[i]:
                if A[i] < B[i]:  # No wrap-around for clockwise path
                    if A[i] < val < B[i]:
                        cw_conflict = True
                        break
                else:  # Wrap-around for clockwise path
                    if val > A[i] or val < B[i]:
                        cw_conflict = True
                        break
        
        # Check if the counterclockwise path has any conflicts
        ccw_conflict = False
        for val in avoid:
            if val != A[i] and val != B[i]:
                if A[i] > B[i]:  # No wrap-around for counter-clockwise path
                    if B[i] < val < A[i]:
                        ccw_conflict = True
                        break
                else:  # Wrap-around for counter-clockwise path
                    if val < A[i] or val > B[i]:
                        ccw_conflict = True
                        break
        
        # Determine the shortest conflict-free path
        if cw_conflict and ccw_conflict:
            return -1
        elif cw_conflict:
            total_ops += ccw_dist
        elif ccw_conflict:
            total_ops += cw_dist
        else:
            total_ops += min(cw_dist, ccw_dist)
    
    return total_ops

# Main function to handle input and output
def main():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Calculate and print the result
    result = min_operations_to_transform(N, M, A, B)
    print(result)

if __name__ == "__main__":
    main()