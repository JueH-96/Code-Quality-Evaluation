def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 2:
        print(abs(A[0] - A[1]))
        return
    
    # Calculate the total maximum score using a greedy approach
    total_score = 0
    
    # We need to maximize the absolute differences, so we should always remove the pair with the largest difference
    # This can be achieved by always considering the largest differences first in a greedy manner
    
    # Calculate differences and sort by their absolute values in descending order
    differences = []
    for i in range(N - 1):
        diff = abs(A[i] - A[i + 1])
        differences.append((diff, i, i + 1))
    
    # Sort differences by absolute value in descending order
    differences.sort(reverse=True, key=lambda x: x[0])
    
    # To keep track of which elements are still in the array
    active = [True] * N
    
    for diff, i, j in differences:
        # Check if both elements are still active
        if active[i] and active[j]:
            # Add the score
            total_score += diff
            # Deactivate the elements
            active[i] = False
            active[j] = False
    
    print(total_score)