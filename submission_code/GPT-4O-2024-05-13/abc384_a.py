# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    c1 = data[1]
    c2 = data[2]
    S = data[3]
    
    result = ''.join([c2 if char != c1 else c1 for char in S])
    print(result)

if __name__ == "__main__":
    main()