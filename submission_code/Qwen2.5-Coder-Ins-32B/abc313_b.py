# YOUR CODE HERE
import sys

def find_strongest_programmer(N, M, relations):
    # Initialize a list to keep track of the number of people each person is stronger than
    stronger_than_count = [0] * (N + 1)
    
    # Initialize a list to keep track of the number of people each person is weaker than
    weaker_than_count = [0] * (N + 1)
    
    # Process each relation
    for A, B in relations:
        stronger_than_count[A] += 1
        weaker_than_count[B] += 1
    
    # Find the person who is stronger than N-1 people and weaker than 0 people
    strongest_programmer = -1
    for i in range(1, N + 1):
        if stronger_than_count[i] == N - 1 and weaker_than_count[i] == 0:
            if strongest_programmer == -1:
                strongest_programmer = i
            else:
                # More than one person satisfies the condition
                return -1
    
    return strongest_programmer

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2
    relations = []
    for _ in range(M):
        A = int(input[index])
        B = int(input[index + 1])
        relations.append((A, B))
        index += 2
    
    result = find_strongest_programmer(N, M, relations)
    print(result)

if __name__ == "__main__":
    main()