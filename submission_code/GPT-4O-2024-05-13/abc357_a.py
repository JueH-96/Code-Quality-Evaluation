# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    H = list(map(int, data[2:2+N]))
    
    count = 0
    for h in H:
        if M >= h:
            count += 1
            M -= h
        else:
            break
    
    print(count)

if __name__ == "__main__":
    main()