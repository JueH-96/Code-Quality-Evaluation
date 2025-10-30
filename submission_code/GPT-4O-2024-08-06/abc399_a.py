# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    hamming_distance = 0
    
    for i in range(N):
        if S[i] != T[i]:
            hamming_distance += 1
    
    print(hamming_distance)

if __name__ == "__main__":
    main()