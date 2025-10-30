# YOUR CODE HERE
def determine_winner(n):
    if n % 3 == 0:
        return "Second"
    elif n % 3 == 1 or n % 3 == 2:
        return "First"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        results.append(determine_winner(n))
    
    print("
".join(results))

if __name__ == "__main__":
    main()