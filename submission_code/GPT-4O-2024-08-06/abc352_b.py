# YOUR CODE HERE
def find_correctly_typed_positions(S, T):
    # Initialize variables
    correct_positions = []
    s_index = 0
    t_index = 0
    
    # Iterate over T and match with S
    while s_index < len(S) and t_index < len(T):
        if S[s_index] == T[t_index]:
            # If characters match, it's a correctly typed character
            correct_positions.append(t_index + 1)  # +1 for 1-based index
            s_index += 1
        t_index += 1
    
    return correct_positions

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    S = data[0]
    T = data[1]
    
    result = find_correctly_typed_positions(S, T)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()