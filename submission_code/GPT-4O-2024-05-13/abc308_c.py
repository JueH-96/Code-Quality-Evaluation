# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    people = []
    
    for i in range(N):
        A = int(data[2 * i + 1])
        B = int(data[2 * i + 2])
        success_rate = A / (A + B)
        people.append((success_rate, i + 1))
    
    people.sort(key=lambda x: (-x[0], x[1]))
    
    result = [str(person[1]) for person in people]
    print(" ".join(result))

if __name__ == "__main__":
    main()