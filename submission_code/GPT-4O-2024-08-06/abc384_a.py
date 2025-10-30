# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    c1 = data[1]
    c2 = data[2]
    S = data[3]
    
    result = []
    for char in S:
        if char == c1:
            result.append(c1)
        else:
            result.append(c2)
    
    print(''.join(result))

if __name__ == "__main__":
    main()