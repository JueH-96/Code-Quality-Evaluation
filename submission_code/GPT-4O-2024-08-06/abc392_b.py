# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = set(int(data[i]) for i in range(2, 2 + M))
    
    missing_numbers = [i for i in range(1, N + 1) if i not in A]
    
    print(len(missing_numbers))
    if missing_numbers:
        print(" ".join(map(str, missing_numbers)))

if __name__ == "__main__":
    main()