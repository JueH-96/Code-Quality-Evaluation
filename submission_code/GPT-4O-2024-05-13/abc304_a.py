# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split('
')
    N = int(data[0])
    people = []
    
    for i in range(1, N + 1):
        name, age = data[i].split()
        age = int(age)
        people.append((name, age, i))
    
    # Sort people by age to find the youngest
    people.sort(key=lambda x: x[1])
    
    # Find the index of the youngest person
    youngest_index = people[0][2] - 1
    
    # Print names in clockwise order starting from the youngest person
    for i in range(N):
        print(data[(youngest_index + i) % N + 1].split()[0])

if __name__ == "__main__":
    main()