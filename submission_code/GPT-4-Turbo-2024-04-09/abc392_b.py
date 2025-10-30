# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    A_set = set(A)
    missing_numbers = []
    
    for i in range(1, N+1):
        if i not in A_set:
            missing_numbers.append(i)
    
    print(len(missing_numbers))
    if missing_numbers:
        print(" ".join(map(str, missing_numbers)))

if __name__ == "__main__":
    main()