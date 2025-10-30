# YOUR CODE HERE
def determine_winner(n):
    if n % 3 == 0:
        return "First"
    else:
        return "Second"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    results = []
    
    for i in range(1, t + 1):
        n = int(data[i])
        results.append(determine_winner(n))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()