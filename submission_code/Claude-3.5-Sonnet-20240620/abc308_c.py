# YOUR CODE HERE
from sys import stdin

def main():
    N = int(stdin.readline())
    people = []
    
    for i in range(1, N + 1):
        A, B = map(int, stdin.readline().split())
        success_rate = A / (A + B)
        people.append((-success_rate, i))
    
    people.sort()
    
    print(' '.join(str(person[1]) for person in people))

if __name__ == "__main__":
    main()