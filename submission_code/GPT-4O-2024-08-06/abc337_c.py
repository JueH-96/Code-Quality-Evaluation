# YOUR CODE HERE
def find_line_order(N, A):
    # Create a list to store the order of people
    order = [0] * N
    
    # Find the person at the front of the line
    front_person = A.index(-1) + 1  # +1 because people are 1-indexed
    order[0] = front_person
    
    # Fill the order list based on the given arrangement
    for i in range(1, N):
        # The next person in line is the one behind the current person
        order[i] = A[order[i-1] - 1]
    
    return order

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = find_line_order(N, A)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()