import sys

def main():
    """
    This function reads the problem inputs, solves the problem, and prints the answer.
    """
    # Read N, X, Y from the first line of standard input
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        N, X, Y = map(int, line1.split())

        # Read the list of sweetness values A
        line2 = sys.stdin.readline()
        if not line2:
            return
        A = list(map(int, line2.split()))

        # Read the list of saltiness values B
        line3 = sys.stdin.readline()
        if not line3:
            return
        B = list(map(int, line3.split()))
    except (IOError, ValueError):
        # This handles cases where input might be malformed or empty.
        return

    # Combine A and B into a list of (sweetness, saltiness) tuples representing dishes
    dishes = list(zip(A, B))

    # --- Strategy 1: Prioritize high sweetness ---
    # Sort dishes by sweetness in descending order
    dishes_sorted_by_A = sorted(dishes, key=lambda x: x[0], reverse=True)
    
    # If no limit is ever exceeded, he eats all N dishes.
    # Initialize the number of dishes for this strategy to N.
    k_A = N
    current_sweetness = 0
    current_saltiness = 0
    for i in range(N):
        current_sweetness += dishes_sorted_by_A[i][0]
        current_saltiness += dishes_sorted_by_A[i][1]
        
        # Check if either limit is exceeded
        if current_sweetness > X or current_saltiness > Y:
            # He stops after eating the (i+1)-th dish
            k_A = i + 1
            break

    # --- Strategy 2: Prioritize high saltiness ---
    # Sort dishes by saltiness in descending order
    dishes_sorted_by_B = sorted(dishes, key=lambda x: x[1], reverse=True)

    # Initialize the number of dishes for this strategy to N.
    k_B = N
    current_sweetness = 0
    current_saltiness = 0
    for i in range(N):
        current_sweetness += dishes_sorted_by_B[i][0]
        current_saltiness += dishes_sorted_by_B[i][1]
        
        # Check if either limit is exceeded
        if current_sweetness > X or current_saltiness > Y:
            # He stops after eating the (i+1)-th dish
            k_B = i + 1
            break
            
    # The answer is the minimum of the outcomes of the two greedy strategies.
    print(min(k_A, k_B))

if __name__ == "__main__":
    main()