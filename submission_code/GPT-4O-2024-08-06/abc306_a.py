# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    result = []
    for char in S:
        result.append(char * 2)
    
    print(''.join(result))

if __name__ == "__main__":
    main()