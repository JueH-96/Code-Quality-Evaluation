# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = data[1]
    
    result = ''.join([char * 2 for char in S])
    print(result)

if __name__ == "__main__":
    main()