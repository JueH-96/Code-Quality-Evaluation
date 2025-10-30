# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    people = []
    
    for i in range(1, N * 2, 2):
        name = data[i]
        age = int(data[i + 1])
        people.append((name, age, (i - 1) // 2))
    
    # Find the index of the youngest person
    youngest_index = min(range(N), key=lambda i: people[i][1])
    
    # Print the names in clockwise order starting from the youngest person
    for i in range(N):
        index = (youngest_index + i) % N
        print(people[index][0])

if __name__ == "__main__":
    main()