# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Calculate the cumulative sum
    cumulative = 0
    min_cumulative = 0
    for a in A:
        cumulative += a
        if cumulative < min_cumulative:
            min_cumulative = cumulative
    
    # The minimum initial passengers is -min_cumulative
    initial = -min_cumulative
    current = initial + cumulative
    
    print(current)

if __name__ == "__main__":
    main()