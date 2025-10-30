# YOUR CODE HERE
import sys

def main():
    import sys
    Q = int(sys.stdin.readline())
    A = []
    for _ in range(Q):
        parts = sys.stdin.readline().strip().split()
        if not parts:
            continue
        if parts[0] == '1':
            x = int(parts[1])
            A.append(x)
        elif parts[0] == '2':
            k = int(parts[1])
            print(A[-k])

if __name__ == "__main__":
    main()