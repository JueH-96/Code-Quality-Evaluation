# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    people = []
    
    index = 1
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        success_rate = A / (A + B)
        people.append((success_rate, i + 1))
        index += 2
    
    # Sort by success rate in descending order, and by person number in ascending order if success rates are the same
    people.sort(key=lambda x: (-x[0], x[1]))
    
    for _, person in people:
        print(person, end=' ')
    print()

if __name__ == "__main__":
    main()