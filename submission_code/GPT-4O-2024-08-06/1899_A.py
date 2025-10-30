# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    results = []
    
    for i in range(1, t + 1):
        n = int(data[i])
        if n % 3 == 0:
            results.append("First")
        else:
            results.append("First")
    
    for result in results:
        print(result)

main()