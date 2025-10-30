def can_color_all(N, K):
    # Check if N is even and K is not half of N
    if N % 2 == 0 and K != N // 2:
        return "Yes"
    # If N is odd, we can always color all points
    return "Yes" if N % 2 == 1 else "No"

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        N, K = map(int, data[i].split())
        results.append(can_color_all(N, K))
    
    print("
".join(results))

if __name__ == "__main__":
    main()