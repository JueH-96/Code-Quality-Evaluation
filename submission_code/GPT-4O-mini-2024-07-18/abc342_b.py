# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    persons = list(map(int, data[1].split()))
    Q = int(data[2])
    
    results = []
    for i in range(3, 3 + Q):
        A, B = map(int, data[i].split())
        # Convert A and B to zero-based index
        index_A = persons.index(A)
        index_B = persons.index(B)
        # Determine which person is further to the front
        if index_A < index_B:
            results.append(A)
        else:
            results.append(B)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()