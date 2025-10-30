# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    people = []
    
    for i in range(1, N + 1):
        A, B = map(int, data[i].split())
        success_rate = A / (A + B)
        people.append((success_rate, i))
    
    # Sort by success rate descending, and by person number ascending
    people.sort(key=lambda x: (-x[0], x[1]))
    
    # Output the sorted person numbers
    result = [str(person[1]) for person in people]
    print(" ".join(result))

if __name__ == "__main__":
    main()