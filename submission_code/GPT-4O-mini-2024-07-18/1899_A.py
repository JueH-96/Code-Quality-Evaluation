def determine_winner(n):
    # Vanya wins if n % 3 == 0 or if n % 3 == 1 (because he can make it divisible by 3)
    # Vova wins if n % 3 == 2 (because Vanya cannot make it divisible by 3 in his turn)
    if n % 3 == 0:
        return "Second"
    else:
        return "First"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    results = []
    
    for i in range(1, t + 1):
        n = int(data[i])
        results.append(determine_winner(n))
    
    print("
".join(results))

if __name__ == "__main__":
    main()