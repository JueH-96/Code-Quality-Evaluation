# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    R = int(data[0])
    G = int(data[1])
    B = int(data[2])
    C = data[3]
    
    if C == "Red":
        # Cannot buy red pen
        result = min(G, B)
    elif C == "Green":
        # Cannot buy green pen
        result = min(R, B)
    elif C == "Blue":
        # Cannot buy blue pen
        result = min(R, G)
    
    print(result)

if __name__ == "__main__":
    main()