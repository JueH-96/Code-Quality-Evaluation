# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    T = list(map(int, data[2:2+Q]))
    
    teeth = [1] * (N + 1)  # Using 1-based indexing
    
    for t in T:
        if teeth[t] == 1:
            teeth[t] = 0
        else:
            teeth[t] = 1
    
    total = sum(teeth[1:])  # Sum from 1 to N
    print(total)

if __name__ == "__main__":
    main()