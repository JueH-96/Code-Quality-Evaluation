# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    results = []
    for i in range(N):
        week_steps = sum(A[i * 7:(i + 1) * 7])
        results.append(week_steps)
    
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()