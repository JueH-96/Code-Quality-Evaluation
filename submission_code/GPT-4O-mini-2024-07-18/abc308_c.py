# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    success_rates = []
    
    for i in range(1, N + 1):
        A, B = map(int, data[i].split())
        success_rate = A / (A + B)
        success_rates.append((success_rate, i))
    
    # Sort by success rate (descending) and then by index (ascending)
    success_rates.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the sorted indices
    result = [str(person[1]) for person in success_rates]
    
    print(" ".join(result))

if __name__ == "__main__":
    main()