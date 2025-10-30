# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = list(data[1])
    Q = int(data[2])
    
    operations = data[3:3+Q]
    
    for operation in operations:
        t, x, c = operation.split()
        t = int(t)
        x = int(x)
        
        if t == 1:
            # Change the x-th character to c
            S[x-1] = c
        elif t == 2:
            # Convert all uppercase letters to lowercase
            S = [ch.lower() for ch in S]
        elif t == 3:
            # Convert all lowercase letters to uppercase
            S = [ch.upper() for ch in S]
    
    print("".join(S))

if __name__ == "__main__":
    main()