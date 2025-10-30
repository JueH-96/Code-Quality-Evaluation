# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    for i in range(N - 1):
        if (S[i] == 'a' and S[i + 1] == 'b') or (S[i] == 'b' and S[i + 1] == 'a'):
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()