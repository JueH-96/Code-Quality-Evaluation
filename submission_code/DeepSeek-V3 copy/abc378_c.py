# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    last_occurrence = {}
    B = []
    
    for i in range(N):
        current = A[i]
        if current in last_occurrence:
            B.append(last_occurrence[current] + 1)  # +1 because positions are 1-based
        else:
            B.append(-1)
        last_occurrence[current] = i  # Update the last occurrence to the current index
    
    print(' '.join(map(str, B)))

if __name__ == "__main__":
    main()